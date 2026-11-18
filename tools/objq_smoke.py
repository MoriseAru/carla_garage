"""Smoke test for use_obj_query_tokens (LOCAL HANDOVER 10947). Read-only w.r.t. runs; one GPU.
--phase off : flag 0 -> forward of tfpp_base_000 on a fixed batch, saved to <out>_off_<tag>.pt (run once with the new fork and
              once with the previous commit's team_code to compare bitwise).
--phase on  : flag 1 -> load tfpp_base_000 non-strict (only obj_query.* missing), shapes, initial loss, gradient reach,
              overfit one batch (matched position error).
"""
import argparse, os, sys, time
import numpy as np, torch
ap = argparse.ArgumentParser()
ap.add_argument("--code", default="/work/gn21/n21001/carla_garage/team_code")
ap.add_argument("--phase", choices=["off", "on"], required=True)
ap.add_argument("--tag", default="new")
ap.add_argument("--src", default="/work/gn21/n21001/carla_garage_runs/tfpp_base_000")
ap.add_argument("--root", default="/work/gn21/n21001/carla_garage_data_root")
ap.add_argument("--out", default="/work/gn21/n21001/V2XState_Real/tmp/objq_smoke")
ap.add_argument("--bs", type=int, default=8); ap.add_argument("--steps", type=int, default=600)
a = ap.parse_args()
sys.path.insert(0, a.code)
import jsonpickle, jsonpickle.ext.numpy as jpn; jpn.register_handlers()  # noqa: E402
import config as cfgmod, model as modmod, data as datamod  # noqa: E402
print("code", a.code, "model.py", modmod.__file__, flush=True)
torch.manual_seed(0); np.random.seed(0); dev = "cuda"
torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

cfg = cfgmod.GlobalConfig(); saved = jsonpickle.decode(open(os.path.join(a.src, "config.json")).read()); cfg.__dict__.update(saved.__dict__)
if a.phase == "on":
    cfg.use_obj_query_tokens = 1; cfg.obj_query_n = 16; cfg.obj_query_radius = 32.0; cfg.obj_query_layers = 4
cfg.initialize(root_dir=[a.root], setting="all")
cfg.augment_percentage = 0.0; cfg.use_color_aug = 0; import random; random.seed(0)   # fixed, un-augmented batch (augment=0 breaks the augmented-file loading)
ds = datamod.CARLA_Data(root=cfg.data_roots, config=cfg, estimate_class_distributions=False, estimate_sem_distribution=False,
                        shared_dict=None, rank=0)
idx = torch.randperm(len(ds), generator=torch.Generator().manual_seed(0))[: a.bs].tolist()
batch = torch.utils.data.default_collate([ds[i] for i in idx])
net = modmod.LidarCenterNet(cfg)
sd = torch.load(os.path.join(a.src, "model_0030.pth"), map_location="cpu")
res = net.load_state_dict(sd, strict=(a.phase == "off"))
print("load:", "strict ok" if a.phase == "off" else f"missing {len(res.missing_keys)} unexpected {len(res.unexpected_keys)}", flush=True)
net.to(dev)


def fwd():
    g = lambda k: batch[k].to(dev, dtype=torch.float32)
    return net(rgb=g("rgb"), lidar_bev=g("lidar"), target_point=g("target_point"), ego_vel=g("speed").unsqueeze(1), command=g("command"))


if a.phase == "off":
    net.eval()
    with torch.no_grad():
        o = fwd()
    keep = {"target_speed": o[1], "checkpoint": o[2]}
    torch.save({k: v.detach().cpu() for k, v in keep.items()}, f"{a.out}_off_{a.tag}.pt")
    print("saved", {k: tuple(v.shape) for k, v in keep.items()}, "sum ts", float(o[1].sum()), "sum ck", float(o[2].sum()), flush=True)
    print("SMOKE_OFF_DONE"); sys.exit(0)

# ---- phase on ----
miss, unexp = res.missing_keys, res.unexpected_keys
assert not unexp and miss and all(k.startswith("obj_query.") for k in miss), (miss[:5], unexp[:5])
n_new = sum(p.numel() for n, p in net.named_parameters() if n.startswith("obj_query."))
print(f"missing keys = obj_query.* only ({len(miss)} tensors, {n_new} params)", flush=True)
from obj_query_tokens import hungarian_loss  # noqa: E402
st, mk = batch["objq_state"].to(dev), batch["objq_mask"].to(dev)
print("labels: objq_state", tuple(st.shape), "valid per frame", mk.sum(1).tolist(), "x,y range (m)",
      float(st[..., :2][mk > 0.5].abs().max()) * 32, "speed max (m/s)", float(st[..., 2][mk > 0.5].max()) * 10, flush=True)
net.train()
o = fwd()
print("memory check: objq tokens", tuple(net._objq["state"].shape), "query memory (H, W, tokens incl. ego)", net.obj_query.last_memory_shape, flush=True)
l_obj, stats = hungarian_loss(net._objq, st, mk)
l_ts = net.loss_speed(o[1], batch["target_speed_twohot"].to(dev, dtype=torch.float32))
print(f"initial: loss_obj_query {float(l_obj):.3f} {stats}; loss_target_speed {float(l_ts):.3f}", flush=True)
# gradient reach: (1) object loss -> queries; (2) planning loss alone -> queries (not detached) and -> planner decoder
net.zero_grad(); l_obj.backward(retain_graph=True)
gq = net.obj_query.query.grad.norm().item(); gj = net.join.layers[0].multihead_attn.in_proj_weight.grad
print(f"grad from loss_obj_query: obj_query.query {gq:.3e}; join (planner decoder) {0.0 if gj is None else gj.norm().item():.3e}", flush=True)
net.zero_grad(); l_ts.backward()
print(f"grad from loss_target_speed: obj_query.query {net.obj_query.query.grad.norm().item():.3e}; obj_query.to_planner "
      f"{net.obj_query.to_planner.weight.grad.norm().item():.3e}; join {net.join.layers[0].multihead_attn.in_proj_weight.grad.norm().item():.3e}", flush=True)
# loss-weight normalisation as train.py would do it (base weights + loss_obj_query 1.0)
w = dict(saved.detailed_loss_weights) if isinstance(getattr(saved, "detailed_loss_weights", None), dict) else None
print("base config detailed_loss_weights (as stored, already normalised):", w, flush=True)
# overfit one batch: train obj_query only, rest frozen (eval for deterministic features)
for n, p in net.named_parameters():
    p.requires_grad_(n.startswith("obj_query."))
net.eval(); net.obj_query.train()
opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=3e-4)
t0 = time.time()
for s in range(1, a.steps + 1):
    fwd(); l, stt = hungarian_loss(net._objq, st, mk)
    opt.zero_grad(); l.backward(); opt.step()
    if s % 50 == 0 or s == 1:
        print(f"overfit step {s} loss {float(l):.4f} pos_err {stt['pos_err_m']:.3f} m l1 {stt['l1']:.4f} bce {stt['bce']:.4f} ({time.time()-t0:.0f}s)", flush=True)
net.obj_query.eval()
with torch.no_grad():
    fwd(); _, stt = hungarian_loss(net._objq, st, mk)
    ex = torch.sigmoid(net._objq["exist_logit"])
print(f"overfit final (eval, no dropout): pos_err {stt['pos_err_m']:.3f} m; exist>0.5 per frame {(ex > 0.5).sum(1).tolist()} vs gt {mk.sum(1).int().tolist()}", flush=True)
print("SMOKE_ON_DONE")
