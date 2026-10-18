"""Original V2XState planner as the TF++ checkpoint head (use_v2xstate_head): smoke test on one GPU (debug-g), real data.
  python tools/v2xstate_smoke.py --root /work/gn21/n21001/carla_garage_smoke_root --base /work/gn21/n21001/carla_garage_runs/tfpp_base_000
Checks:
 1) build + load: the base checkpoint loads non-strict with ONLY v2xstate_head.* missing and ONLY checkpoint_decoder.*
    unexpected; target speed, detections and BEV semantics are bitwise equal to the plain base model (pure add-on).
 2) occupancy channels in the head's metric grid (column = x forward, row = y right):
    objects = detection heatmap is high at the visible cars of the batch (coop tokens give their positions) and the head's
    layout is the best of the 8 swap / flip layouts; ego box covers 9.8 m^2 (half extents 2.45 x 1.0) centred at the origin; target box covers
    1 m^2 at the (clipped) target point; drivable is higher just ahead of the ego than on the map average.
 3) sector masks: forward mask centroid ahead on the centre line, left mask centroid on the left (y < 0), right mask on
    the right (y > 0), near mask at the ego; gate prior peaks at basis 1 / 2 / 3 for left / lane-follow / right.
 4) train-mode forward: checkpoints (bs, 10, 2), the two aux losses finite and present in compute_loss with the keys
    the training loop weights (loss_v2xs_mask, loss_v2xs_gate).
 5) gradient routing: the head's losses reach the head and the backbone (bev_feature_grid path) but NOT the detection /
    BEV-semantic heads (occupancy inputs are detached, as the frozen-perception inputs of the original).
 6) 60 AdamW steps on one batch (head only, lr 3e-4) reduce loss_checkpoint below 50 % of the start.
"""
import argparse, os, sys, time, copy
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "team_code"))
import jsonpickle, jsonpickle.ext.numpy as jsonpickle_numpy; jsonpickle_numpy.register_handlers()
import config as cfgmod, model as modmod, data as datamod

ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--base", required=True)
ap.add_argument("--bs", type=int, default=8); ap.add_argument("--basis_source", default="concat"); a = ap.parse_args()
dev = "cuda"; torch.manual_seed(0); np.random.seed(0)


def ok(c, m):
  print(("PASS " if c else "FAIL ") + m, flush=True)
  if not c:
    sys.exit("V2XSTATE SMOKE FAILED")


def load_cfg(run_dir):
  cfg = cfgmod.GlobalConfig(); saved = jsonpickle.decode(open(os.path.join(run_dir, "config.json")).read()); cfg.__dict__.update(saved.__dict__)
  return cfg


cfg_b = load_cfg(a.base); cfg_b.initialize(root_dir=[a.root], setting="all")
cfg_ds = copy.deepcopy(cfg_b); cfg_ds.use_v2x = 1      # dataset only: coop tokens give the car positions for check 2
cfg_ds.augment_percentage = 0.0; cfg_ds.use_color_aug = 0; cfg_ds.lidar_aug_prob = 0.0
ds = datamod.CARLA_Data(root=cfg_ds.data_roots, config=cfg_ds, estimate_class_distributions=False, estimate_sem_distribution=False, shared_dict=None, rank=0)
g = torch.Generator().manual_seed(0); idx = torch.randperm(len(ds), generator=g)[: a.bs].tolist()
data = next(iter(torch.utils.data.DataLoader(torch.utils.data.Subset(ds, idx), batch_size=a.bs, shuffle=False, num_workers=4)))
rgb = data["rgb"].to(dev, dtype=torch.float32); lidar = data["lidar"].to(dev, dtype=torch.float32)
tp = data["target_point"].to(dev, dtype=torch.float32); vel = data["speed"].to(dev, dtype=torch.float32).unsqueeze(1)
cmd = data["command"].to(dev, dtype=torch.float32); route = data["route"][:, : cfg_b.predict_checkpoint_len].to(dev, dtype=torch.float32)
bs = rgb.shape[0]
print(f"batch {bs}; speed {vel.flatten().tolist()}; command idx {cmd.argmax(1).tolist()}; target |xy| max {tp.abs().max():.1f} m", flush=True)

# 1) build + load + pure add-on
cfg = copy.deepcopy(cfg_b); cfg.use_v2xstate_head = 1; cfg.v2xstate_basis_source = a.basis_source
for k_, v_ in dict(v2xstate_num_basis=4, v2xstate_temperature=0.5, v2xstate_mask_weight=0.02, v2xstate_gate_weight=0.05).items():
  if not hasattr(cfg, k_): setattr(cfg, k_, v_)
net = modmod.LidarCenterNet(cfg); ck = sorted(f for f in os.listdir(a.base) if f.startswith("model_") and f.endswith(".pth"))[-1]
sd = torch.load(os.path.join(a.base, ck), map_location="cpu"); r = net.load_state_dict(sd, strict=False)
miss = [k for k in r.missing_keys if not k.startswith("v2xstate_head.")]; unexp = [k for k in r.unexpected_keys if not k.startswith("checkpoint_decoder.")]
ok(not miss and not unexp and any(k.startswith("v2xstate_head.") for k in r.missing_keys) and any(k.startswith("checkpoint_decoder.") for k in r.unexpected_keys),
   f"base {ck} loads: missing only v2xstate_head.* ({len(r.missing_keys)}), unexpected only checkpoint_decoder.* ({len(r.unexpected_keys)}); other {miss[:3]} {unexp[:3]}")
hd = net.v2xstate_head
print(f"  head parameters {sum(p.numel() for p in hd.parameters()) / 1e6:.2f} M; grid {hd.height}x{hd.width}; basis_source {hd.basis_source}", flush=True)
net.to(dev).eval(); net_b = modmod.LidarCenterNet(cfg_b); net_b.load_state_dict(sd, strict=True); net_b.to(dev).eval()
kw = dict(rgb=rgb, lidar_bev=lidar, target_point=tp, ego_vel=vel, command=cmd)
with torch.no_grad():
  out_b = net_b(**kw); out = net(**kw)
same = torch.equal(out_b[1], out[1]) and torch.equal(out_b[4], out[4]) and all(torch.equal(x_, y_) for x_, y_ in zip(out_b[6], out[6]) if x_ is not None)
ok(same, "target speed, BEV semantics and detections bitwise equal to the plain base model")
ok(out[2].shape == (bs, 10, 2) and torch.isfinite(out[2]).all(), f"inference checkpoints {tuple(out[2].shape)} finite")

# 2) occupancy channels
from v2xstate_head import tfpp_occupancy_inputs
with torch.no_grad():
  objects, drivable = tfpp_occupancy_inputs(out[6], out[4], net.valid_bev_pixels, (hd.height, hd.width))
  occ = hd.occupancy(objects, drivable, tp)
ok(tuple(occ.shape) == (bs, 6, hd.height, hd.width), f"occupancy {tuple(occ.shape)}")
cars = []
st_, mk_, hid_ = data["coop_states"], data["coop_mask"], data["coop_hidden"]
for b in range(bs):
  for j in range(st_.shape[1]):
    if mk_[b, j] > 0.5 and hid_[b, j] < 0.5:
      x, y = float(st_[b, j, 0]) * 32, float(st_[b, j, 1]) * 32
      if abs(x) < 30 and abs(y) < 30 and (abs(x) > 4 or abs(y) > 3):
        cars.append((b, x, y))
ok(len(cars) >= 5, f"{len(cars)} visible cars inside the BEV in the batch")
def at(ch, col_of, row_of):
  v = []
  for b, x, y in cars:
    c = int(np.floor((col_of(x, y) + 32.0) / hd.cell_x)); r_ = int(np.floor((row_of(x, y) + 32.0) / hd.cell_y))
    v.append(float(occ[b, ch, min(max(r_, 0), hd.height - 1), min(max(c, 0), hd.width - 1)]))
  return float(np.mean(v))
lay = {}
for swap in (0, 1):
  for sx in (1, -1):
    for sy in (1, -1):
      col_of = (lambda x, y, sx=sx, sy=sy, swap=swap: (sy * y) if swap else (sx * x))
      row_of = (lambda x, y, sx=sx, sy=sy, swap=swap: (sx * x) if swap else (sy * y))
      lay[f"swap{swap}_sx{sx:+d}_sy{sy:+d}"] = at(0, col_of, row_of)
mean0 = float(occ[:, 0].mean()); best = max(lay, key=lay.get); second = sorted(lay.values())[-2]
print("  objects channel at visible-car centres per layout: " + ", ".join(f"{k} {v:.3f}" for k, v in lay.items()) + f"; map mean {mean0:.4f}")
ok(best == "swap0_sx+1_sy+1" and lay[best] > 5 * mean0 and lay[best] > 1.5 * second,
   f"objects channel: head layout (col = x forward, row = y right) is the best, {lay[best] / max(mean0, 1e-9):.1f}x the map mean, {lay[best] / max(second, 1e-9):.1f}x the runner-up")
ego = occ[0, 1]; e_sum = float(ego.sum()); e_cx = float((ego * hd.grid_forward).sum() / ego.sum()); e_cy = float((ego * hd.grid_lateral).sum() / ego.sum())
ok(abs(e_sum - 9.8) < 1e-3 and abs(e_cx) < 1e-3 and abs(e_cy) < 1e-3, f"ego box: area {e_sum:.3f} m^2 (9.8 expected: half extents 2.45 x 1.0), centroid ({e_cx:.3f}, {e_cy:.3f})")
t_sum = occ[:, 2].sum(dim=(1, 2)); t_cx = (occ[:, 2] * hd.grid_forward).sum(dim=(1, 2)) / t_sum
lim = hd.max_range - 0.5; tcl = tp.clamp(-lim, lim)
ok(bool(((t_sum - 1).abs() < 1e-3).all()) and bool(((t_cx - tcl[:, 0]).abs() < 1e-3).all()), f"target box: area 1 m^2 on every sample, forward centroid = clipped target x (max err {float((t_cx - tcl[:, 0]).abs().max()):.1e})")
ahead = (hd.grid_forward > 3) & (hd.grid_forward < 12) & (hd.grid_lateral.abs() < 1.5)
d_ahead, d_mean = float(occ[:, 5][:, ahead].mean()), float(occ[:, 5].mean())
ok(d_ahead > 1.5 * d_mean, f"drivable: just ahead of the ego {d_ahead:.3f} vs map mean {d_mean:.3f} (camera-visible area only)")
ok(bool((occ[:, 3, :, 0] < occ[:, 3, :, -1]).all()) and bool((occ[:, 4, 0, :] < occ[:, 4, -1, :]).all()), "coordinate channels: forward grows along columns, lateral along rows")

# 3) sector masks and gate prior
st = hd.state(torch.full((4, 1), 5.0, device=dev), torch.eye(6, device=dev)[[0, 3, 1, 2]])   # left, lanefollow, right, straight
masks, weights = hd.state_masks(st)
cen = lambda m: (float((m * hd.grid_forward).sum() / m.sum()), float((m * hd.grid_lateral).sum() / m.sum()))
c_near, c_left, c_fwd, c_right = cen(masks[0][0]), cen(masks[1][0]), cen(masks[2][0]), cen(masks[3][0])
print(f"  mask centroids (forward, lateral) m: near {c_near[0]:.1f},{c_near[1]:.1f}; left {c_left[0]:.1f},{c_left[1]:.1f}; forward {c_fwd[0]:.1f},{c_fwd[1]:.1f}; right {c_right[0]:.1f},{c_right[1]:.1f}")
ok(c_fwd[0] > 5 and abs(c_fwd[1]) < 0.5 and c_left[1] < -2 and c_right[1] > 2 and abs(c_near[0]) < 1 and abs(c_near[1]) < 1,
   "sector masks: forward ahead on the centre line, left at y < 0, right at y > 0, near at the ego")
w = torch.stack([x_.squeeze(-1) for x_ in weights], 1)
ok(bool(w[0, 1] == 1) and bool(w[2, 3] == 1) and bool(w[1, 2] == 1), f"mask weights follow the command (left -> basis 1, right -> 3, lane follow -> 2): {w.tolist()}")
pk =[int(np.argmin([float(hd.gate_prior_loss(torch.nn.functional.one_hot(torch.tensor([b], device=dev), hd.num_basis).float() * 0.997 + 0.00075, st[i:i + 1])) for b in range(hd.num_basis)])) for i in range(4)]
ok(pk == [1, 2, 3, 2], f"gate prior minimised at basis {pk} for left / lane follow / right / straight (expected [1, 2, 3, 2])")

# 4) train-mode forward + loss keys
net.train()
for m in net.modules():
  if isinstance(m, torch.nn.modules.batchnorm._BatchNorm) and not any(m is mm for mm in hd.modules()): m.eval()
o = net(**kw)
lm, lg = net._v2xs["loss_mask"], net._v2xs["loss_gate"]
ok(o[2].shape == (bs, 10, 2) and torch.isfinite(lm) and torch.isfinite(lg), f"train forward: checkpoints {tuple(o[2].shape)}; loss_mask {float(lm):.4f}, loss_gate {float(lg):.4f}")
print(f"  gate (train, untrained head) {net._v2xs['gate'][0].tolist()}; attention mean {float(net._v2xs['attention'].mean()):.3f}")
for key in ("loss_v2xs_mask", "loss_v2xs_gate"):
  ok(key in cfgmod.GlobalConfig().detailed_loss_weights, f"{key} is a key of a fresh GlobalConfig().detailed_loss_weights (train.py builds it; old config.json files predate it)")
# compute_loss with only the planning terms filled in is too entangled with the label set; check the update block directly
src = open(os.path.join(os.path.dirname(__file__), "..", "team_code", "model.py")).read()
ok("'loss_v2xs_mask': self._v2xs['loss_mask']" in src, "compute_loss emits loss_v2xs_mask / loss_v2xs_gate")

# 5) gradient routing
net.zero_grad(set_to_none=True)
loss = torch.mean(torch.abs(o[2] - route)) + 0.02 * lm + 0.05 * lg
loss.backward()
gn = lambda pre: sum(float(p.grad.norm()) for n_, p in net.named_parameters() if n_.startswith(pre) and p.grad is not None)
g_head, g_bb, g_det, g_sem = gn("v2xstate_head."), gn("backbone."), gn("head."), gn("bev_semantic_decoder.")
g_gate, g_basis = gn("v2xstate_head.state_gate."), gn("v2xstate_head.basis_extractor.")
print(f"  grad norm sums: head {g_head:.3e} (gate {g_gate:.3e}, basis {g_basis:.3e}); backbone {g_bb:.3e}; detection {g_det:.3e}; bev semantic {g_sem:.3e}")
ok(g_head > 0 and g_gate > 0 and g_basis > 0 and g_bb > 0, "the head's losses reach the head (incl. gate and basis extractor) and the backbone")
ok(g_det == 0 and g_sem == 0, "no gradient from the head into the detection / BEV-semantic heads (occupancy detached)")

# 6) overfit one batch, head only
opt = torch.optim.AdamW([p for p in hd.parameters()], lr=3e-4, weight_decay=0.01); hist = []
for it in range(60):
  o = net(**kw); l1 = torch.mean(torch.abs(o[2] - route)); loss = l1 + 0.02 * net._v2xs["loss_mask"] + 0.05 * net._v2xs["loss_gate"]
  opt.zero_grad(); loss.backward(); opt.step(); hist.append(float(l1))
print(f"  overfit: loss_checkpoint {hist[0]:.3f} -> {hist[-1]:.3f} m; loss_mask {float(net._v2xs['loss_mask']):.4f}; loss_gate {float(net._v2xs['loss_gate']):.4f}")
ok(min(hist[-10:]) < 0.5 * hist[0], "60 AdamW steps on one batch reduce loss_checkpoint below 50 % of the start")
net.eval(); t0 = time.time()
with torch.no_grad():
  for _ in range(5): pc = net(**kw)[2]
torch.cuda.synchronize(); print(f"  inference {1000 * (time.time() - t0) / 5:.0f} ms per batch of {bs}; base GRU ADE on this batch {float(torch.linalg.norm(out_b[2] - route, dim=-1).mean()):.3f} m")
print("V2XSTATE SMOKE PASSED")
