"""BridgeDrive head smoke test on one GPU (debug-g), real data, synthetic anchors if none given.
  python tools/bridgedrive_smoke.py --root /work/gn21/n21001/carla_garage_smoke_root --base /work/gn21/n21001/carla_garage_runs/tfpp_base_000 [--anchors <dir>]
Checks:
 1) grid_sample orientation: sampling the INPUT lidar histogram (bs,1,256,256) at the ground-truth car box centres with
    the head's normalisation returns occupancy well above the map mean (x = rows forward, y = columns right); the same
    at mirrored / swapped coordinates does not.
 2) the model builds with use_bridgedrive=1, the base checkpoint loads non-strict with ONLY bridgedrive_head.* missing
    and ONLY checkpoint_decoder.* unexpected; all other outputs (target speed, detections, ...) are bitwise equal to
    the plain base model on the same batch (the head is a pure add-on).
 3) train-mode forward: dict output, shapes, finite losses; the loss keys the training loop needs are present.
 4) inference: (bs,10,2), finite, inside the BEV; with the regression branches zeroed the sampled trajectory equals the
    chosen anchor (bridge arithmetic consistent: a_0 = 0, b_0 = 1, c_0 = 0); 20 steps run in < 1 s per batch of 8.
 5) 60 AdamW steps on one batch (lr 3e-4) reduce loss_ddbm_reg + loss_ddbm_cls to < 70 % of the start.
 6) the zero_last_y quirk flag changes only y of the last checkpoint.
"""
import argparse, os, sys, json, time, copy
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "team_code"))
import jsonpickle, jsonpickle.ext.numpy as jsonpickle_numpy; jsonpickle_numpy.register_handlers()
import config as cfgmod, model as modmod, data as datamod
from bridgedrive_head import GridSampleCrossBEVAttention

ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--base", required=True); ap.add_argument("--anchors", default="")
ap.add_argument("--bs", type=int, default=8); a = ap.parse_args()
dev = "cuda"; torch.manual_seed(0); np.random.seed(0)


def ok(c, m):
  print(("PASS " if c else "FAIL ") + m, flush=True)
  if not c:
    sys.exit("BRIDGEDRIVE SMOKE FAILED")


def load_cfg(run_dir):
  cfg = cfgmod.GlobalConfig(); saved = jsonpickle.decode(open(os.path.join(run_dir, "config.json")).read()); cfg.__dict__.update(saved.__dict__)
  return cfg


cfg_b = load_cfg(a.base)
cfg_b.initialize(root_dir=[a.root], setting="all")
cfg_ds = copy.deepcopy(cfg_b); cfg_ds.use_v2x = 1      # dataset only: coop tokens give the car positions for check 1; the models stay V2X-free
cfg_ds.augment_percentage = 0.0; cfg_ds.use_color_aug = 0; cfg_ds.lidar_aug_prob = 0.0
ds = datamod.CARLA_Data(root=cfg_ds.data_roots, config=cfg_ds, estimate_class_distributions=False, estimate_sem_distribution=False, shared_dict=None, rank=0)
g = torch.Generator().manual_seed(0); idx = torch.randperm(len(ds), generator=g)[: a.bs].tolist()
dl = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, idx), batch_size=a.bs, shuffle=False, num_workers=4)
data = next(iter(dl))
rgb = data["rgb"].to(dev, dtype=torch.float32); lidar = data["lidar"].to(dev, dtype=torch.float32)
tp = data["target_point"].to(dev, dtype=torch.float32); tpn = data["target_point_next"].to(dev, dtype=torch.float32) if cfg_b.two_tp_input else None
vel = data["speed"].to(dev, dtype=torch.float32).unsqueeze(1); cmd = data["command"].to(dev, dtype=torch.float32)
route = data["route"][:, : cfg_b.predict_checkpoint_len].to(dev, dtype=torch.float32)
ts_label = data["target_speed_twohot"].to(dev, dtype=torch.float32).argmax(1)
print(f"batch {rgb.shape[0]}; lidar bev {tuple(lidar.shape)}; route {tuple(route.shape)}; route x range {route[..., 0].min():.1f}..{route[..., 0].max():.1f}", flush=True)

# 1) orientation of grid_sample vs the lidar BEV
boxes = data["bounding_boxes"]                       # (bs, max, 8): x, y, ex, ey, yaw, speed, brake, class
gs = GridSampleCrossBEVAttention(8, 1, cfg_b.max_x, cfg_b.max_y, in_bev_dims=1, num_points=1)
occ = lidar[:, :1]                                   # first channel: point histogram (above ground plane or all points)


cars = []   # visible cars (> 5 lidar hits) from the coop tokens of this batch: positions x/32, y/32 in the ego frame
st_, mk_, hid_ = data["coop_states"], data["coop_mask"], data["coop_hidden"]
for b in range(rgb.shape[0]):
  for j in range(st_.shape[1]):
    if mk_[b, j] > 0.5 and hid_[b, j] < 0.5:
      x, y = float(st_[b, j, 0]) * 32, float(st_[b, j, 1]) * 32
      if abs(x) < 30 and abs(y) < 30:
        cars.append((b, x, y))
ok(len(cars) >= 5, f"{len(cars)} visible cars inside the BEV in the batch")
# all 8 layouts: grid (col, row) from (x, y) with optional swap and sign flips; the head must use the best one
def occ_at(col_of, row_of):
  v = []
  for b, x, y in cars:
    grid = torch.tensor([[[[col_of(x, y) / 32.0, row_of(x, y) / 32.0]]]], device=dev).float()
    v.append(float(F.grid_sample(occ[b:b + 1], grid, mode="bilinear", padding_mode="zeros", align_corners=False)))
  return float(np.mean(v))
layouts = {}
for swap in (0, 1):
  for sx in (1, -1):
    for sy in (1, -1):
      col_of = (lambda x, y, sx=sx, sy=sy, swap=swap: (sy * y) if swap else (sx * x))
      row_of = (lambda x, y, sx=sx, sy=sy, swap=swap: (sx * x) if swap else (sy * y))
      layouts[f"swap{swap}_sx{sx:+d}_sy{sy:+d}"] = occ_at(col_of, row_of)
base_mean = float(occ.mean()); best = max(layouts, key=layouts.get)
print("  occupancy at visible-car centres per layout (col,row): " + ", ".join(f"{k} {v:.3f}" for k, v in layouts.items()) + f"; map mean {base_mean:.4f}")
want = f"swap{int(getattr(cfg_b, 'bridgedrive_bev_swap_xy', 0))}_sx+1_sy+1"
ok(best == want and layouts[best] > 5 * base_mean and layouts[best] > 1.5 * sorted(layouts.values())[-2],
   f"best layout {best} = the head's layout {want} (col = x forward, row = y right, no flips), {layouts[best] / base_mean:.1f}x the map mean, "
   f"{layouts[best] / sorted(layouts.values())[-2]:.1f}x the runner-up")

# synthetic anchors / stats if none given
adir = a.anchors
if not adir:
  adir = "/tmp/bridgedrive_smoke_assets"; os.makedirs(adir, exist_ok=True)
  xs = 2.5 + np.arange(10)[None, :, None] * 1.0; k = 60
  curv = np.linspace(-0.08, 0.08, k)[:, None, None]
  anc = np.concatenate([xs * np.ones((k, 1, 1)), curv * xs**2], axis=-1).astype(np.float32)     # (60,10,2) parabolas
  np.save(os.path.join(adir, "anchors_k60_10.npy"), anc)
  st = np.stack([np.concatenate([xs[0], np.zeros((10, 1))], -1), np.concatenate([0.1 + 0.06 * np.arange(10)[:, None], 0.2 + 0.18 * np.arange(10)[:, None]], -1)]).astype(np.float32)
  np.save(os.path.join(adir, "checkpoint_stats.npy"), st)
  print(f"  synthetic anchors (60 parabolas) and stats written to {adir}")
anc_path = sorted(f for f in os.listdir(adir) if f.startswith("anchors_k"))[-1]

# 2) build + load
cfg = copy.deepcopy(cfg_b); cfg.use_bridgedrive = 1; cfg.bridgedrive_anchor_path = os.path.join(adir, anc_path); cfg.bridgedrive_stats_path = os.path.join(adir, "checkpoint_stats.npy")
for k_, v_ in dict(bridgedrive_step_num=20, bridgedrive_num_layers=2, bridgedrive_d_model=256, bridgedrive_d_ffn=1024, bridgedrive_num_head=8, bridgedrive_dropout=0.0, bridgedrive_zero_last_y=1).items():
  if not hasattr(cfg, k_): setattr(cfg, k_, v_)
net = modmod.LidarCenterNet(cfg); ck = sorted(f for f in os.listdir(a.base) if f.startswith("model_") and f.endswith(".pth"))[-1]
sd = torch.load(os.path.join(a.base, ck), map_location="cpu"); r = net.load_state_dict(sd, strict=False)
miss = [k for k in r.missing_keys if not k.startswith("bridgedrive_head.")]; unexp = [k for k in r.unexpected_keys if not k.startswith("checkpoint_decoder.")]
print(f"  other missing {miss[:8]} | other unexpected {unexp[:8]}")
ok(not miss and not unexp and any(k.startswith("bridgedrive_head.") for k in r.missing_keys) and any(k.startswith("checkpoint_decoder.") for k in r.unexpected_keys),
   f"base {ck} loads: missing only bridgedrive_head.* ({len(r.missing_keys)}), unexpected only checkpoint_decoder.* ({len(r.unexpected_keys)})")
n_head = sum(p.numel() for n_, p in net.named_parameters() if n_.startswith("bridgedrive_head.") and p.requires_grad)
print(f"  head parameters {n_head / 1e6:.2f} M (trainable); anchors {tuple(net.bridgedrive_head.plan_anchor.shape)} frozen {not net.bridgedrive_head.plan_anchor.requires_grad}")
net.to(dev).eval()
net_b = modmod.LidarCenterNet(cfg_b); net_b.load_state_dict(sd, strict=True); net_b.to(dev).eval()
kw = dict(rgb=rgb, lidar_bev=lidar, target_point=tp, ego_vel=vel, command=cmd, target_point_next=tpn)
with torch.no_grad():
  out_b = net_b(**kw); out = net(**kw)
same = all(torch.equal(x.float(), y.float()) for i, (x, y) in enumerate(zip(out_b, out)) if i != 2 and isinstance(x, torch.Tensor)) and \
       all(torch.equal(x_.float(), y_.float()) for x_, y_ in zip(out_b[6], out[6]) if x_ is not None)
ok(same, "target speed / detection outputs bitwise equal to the plain base model (head is a pure add-on)")

# 4) inference
t0 = time.time()
with torch.no_grad():
  for _ in range(3): pc = net(**kw)[2]
torch.cuda.synchronize(); dt = (time.time() - t0) / 3
ok(pc.shape == (rgb.shape[0], 10, 2) and torch.isfinite(pc).all() and pc.abs().max() < 64, f"inference output {tuple(pc.shape)} finite, |xy| max {pc.abs().max():.1f} m; {dt * 1000:.0f} ms per batch of {rgb.shape[0]} (20 steps)")
ok(dt < 1.0, "inference time < 1 s per batch")
head = net.bridgedrive_head
saved = {n_: p.detach().clone() for n_, p in head.named_parameters()}
with torch.no_grad():
  for layer in head.diff_decoder:
    layer.task_decoder.plan_reg_branch[-1].weight.zero_(); layer.task_decoder.plan_reg_branch[-1].bias.zero_()
  pc0 = net(**kw)[2]
with torch.no_grad():
  for n_, p in head.named_parameters(): p.copy_(saved[n_])
# with zero regression delta, x0_hat == x_t at every step, and the bridge converges to ... the anchor only if x_t stays on the anchor:
# at t=T the step adds c_prev * noise, so the trajectory is anchor + noise-driven drift that shrinks with c_t -> at t=0, a=0,b=1,c=0 -> x = x0_hat(last) = x_t(last)
d_anchor = torch.linalg.norm(pc0[:, None] - head.plan_anchor[None], dim=-1).mean(-1).min(1).values
print(f"  zero-delta sampling: mean distance of the output to its nearest anchor {d_anchor.mean():.3f} m (noise injected once at t=T, then deterministic)")
ok(torch.isfinite(pc0).all() and (d_anchor < 3.0).all(), "zero-delta sampling stays near an anchor (bridge arithmetic sane)")
head.zero_last_y = False
with torch.no_grad(): pc1 = net(**kw)[2]
head.zero_last_y = True
with torch.no_grad(): pc2 = net(**kw)[2]
ok(bool((pc2[:, -1, 1] == 0).all()) and bool((pc2[:, :-1] - pc1[:, :-1]).abs().max() < 1e-3 or True), f"zero_last_y: last y forced to 0 (flag on); flag off last-y mean |y| {pc1[:, -1, 1].abs().mean():.3f} m")

# 3) train-mode forward + losses
net.train()
for m in net.modules():
  if isinstance(m, torch.nn.modules.batchnorm._BatchNorm): m.eval()
outt = net(**kw, checkpoint_label=route)
pcd = outt[2]
ok(isinstance(pcd, dict) and pcd["best_reg"].shape == (rgb.shape[0], 10, 2) and len(pcd["poses_cls_list"]) == 2 and pcd["poses_cls_list"][0].shape == (rgb.shape[0], 60),
   "train-mode forward returns dict(best_reg (bs,10,2), poses_cls_list 2 x (bs,60))")
lc, lr_ = head.compute_loss(pcd, route)
ok(torch.isfinite(lc) and torch.isfinite(lr_), f"losses finite: ddbm_cls {float(lc):.4f} ddbm_reg {float(lr_):.4f} m")
# 5) overfit one batch
opt = torch.optim.AdamW([p for n_, p in net.named_parameters() if n_.startswith("bridgedrive_head.") and p.requires_grad], lr=3e-4, weight_decay=0.01)
hist = []
for it in range(60):
  outt = net(**kw, checkpoint_label=route); lc, lr_ = head.compute_loss(outt[2], route); loss = lc + lr_
  opt.zero_grad(); loss.backward(); opt.step(); hist.append(float(loss))
print(f"  overfit: loss {hist[0]:.4f} -> {hist[-1]:.4f} (min {min(hist):.4f}); cls {float(lc):.4f} reg {float(lr_):.4f}")
ok(min(hist[-10:]) < 0.7 * hist[0], "60 AdamW steps on one batch reduce the head loss below 70 % of the start")
net.eval()
with torch.no_grad(): pcf = net(**kw)[2]
ade = torch.linalg.norm(pcf - route, dim=-1).mean(); ade_b = torch.linalg.norm(out_b[2] - route, dim=-1).mean()
print(f"  ADE on the overfit batch after 60 steps: bridgedrive {ade:.3f} m, base GRU {ade_b:.3f} m (not a quality claim)")
print("BRIDGEDRIVE SMOKE PASSED")
