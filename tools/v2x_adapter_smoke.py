"""Structural checks for the frozen-base residual adapter (scheme A, model.V2XResidualAdapter).
  python tools/v2x_adapter_smoke.py --root /work/gn21/n21001/carla_garage_smoke_root --base /work/gn21/n21001/carla_garage_runs/tfpp_base_000
"""
import argparse, os, sys, copy
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "team_code"))
import jsonpickle, jsonpickle.ext.numpy as jsonpickle_numpy; jsonpickle_numpy.register_handlers()
import config as cfgmod, model as modmod, data as datamod, v2x_features

ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--base", required=True); a = ap.parse_args()
dev = "cuda"; torch.manual_seed(0); np.random.seed(0)
def ok(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg, flush=True)
    if not cond: sys.exit("ADAPTER SMOKE FAILED")

def base_cfg():
    cfg = cfgmod.GlobalConfig(); saved = jsonpickle.decode(open(os.path.join(a.base, "config.json")).read()); cfg.__dict__.update(saved.__dict__); return cfg
cfg_b = base_cfg(); cfg_a = base_cfg()
cfg_a.use_v2x = 1; cfg_a.use_v2x_adapter = 1; cfg_a.v2x_adapter_layers = 2; cfg_a.v2x_adapter_heads = 8; cfg_a.v2x_adapter_ffn = 512
cfg_a.initialize(root_dir=[a.root], setting="all", use_v2x=1)
cfg_a.augment_percentage = 0.0; cfg_a.use_color_aug = 0; cfg_a.lidar_aug_prob = 0.0   # deterministic samples (feature caching)
ds = datamod.CARLA_Data(root=cfg_a.data_roots, config=cfg_a, estimate_class_distributions=False, estimate_sem_distribution=False, shared_dict=None, rank=0)
idx = list(range(0, len(ds), max(1, len(ds) // 8)))[:8]
dl = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, idx), batch_size=8, shuffle=False, num_workers=0)
data = next(iter(dl)); data2 = next(iter(torch.utils.data.DataLoader(torch.utils.data.Subset(ds, idx), batch_size=8, shuffle=False, num_workers=0)))
ok(torch.equal(data["rgb"], data2["rgb"]) and torch.equal(data["lidar"], data2["lidar"]) and torch.equal(data["coop_states"], data2["coop_states"]),
   "deterministic loader (augment_percentage=0, color aug off): identical rgb / lidar / coop states on reload")

sd_base = torch.load(sorted(f for f in os.listdir(a.base) if f.startswith("model_"))[-1].join([a.base + "/", ""]), map_location="cpu")
net_b = modmod.LidarCenterNet(cfg_b); net_b.load_state_dict(sd_base, strict=True); net_b.to(dev).eval()
net_a = modmod.LidarCenterNet(cfg_a); res = net_a.load_state_dict(sd_base, strict=False); net_a.to(dev).eval()
ok(len(res.unexpected_keys) == 0 and all(k.startswith("v2x_adapter.") for k in res.missing_keys) and len(res.missing_keys) > 0,
   f"base weights load into the adapter model: missing = adapter params only ({len(res.missing_keys)} tensors), unexpected 0")
n_ad = sum(p.numel() for n, p in net_a.named_parameters() if n.startswith("v2x_adapter.")); n_all = sum(p.numel() for p in net_a.parameters())
print(f"adapter params {n_ad/1e6:.2f}M of {n_all/1e6:.1f}M ({100*n_ad/n_all:.2f}%)")
ok(not hasattr(net_a, "coop_proj") and hasattr(net_a, "v2x_adapter"), "adapter model has no memory-token modules (coop_proj) and has v2x_adapter")
groups = net_a.create_optimizer_groups(0.01); ok(sum(len(g["params"]) for g in groups) == len(list(net_a.parameters())), "create_optimizer_groups classifies every parameter (incl. adapter)")

rgb = data["rgb"].to(dev, dtype=torch.float32); lidar = data["lidar"].to(dev, dtype=torch.float32); tp = data["target_point"].to(dev, dtype=torch.float32)
vel = data["speed"].to(dev, dtype=torch.float32).unsqueeze(1); cmd = data["command"].to(dev, dtype=torch.float32)
st = data["coop_states"].to(dev, dtype=torch.float32); mk = data["coop_mask"].to(dev, dtype=torch.float32)
ok(float(mk.sum()) > 0, f"batch has valid coop tokens ({int(mk.sum())} slots over {mk.shape[0]} frames)")
with torch.no_grad():
    ob = net_b(rgb=rgb, lidar_bev=lidar, target_point=tp, ego_vel=vel, command=cmd)
    oa0 = net_a(rgb=rgb, lidar_bev=lidar, target_point=tp, ego_vel=vel, command=cmd, coop_states=st, coop_mask=torch.zeros_like(mk))
    oa1 = net_a(rgb=rgb, lidar_bev=lidar, target_point=tp, ego_vel=vel, command=cmd, coop_states=st, coop_mask=mk)
    oan = net_a(rgb=rgb, lidar_bev=lidar, target_point=tp, ego_vel=vel, command=cmd)
ok(torch.equal(oa0[1], ob[1]) and torch.equal(oa0[2], ob[2]), "rate 0 (all slots masked) == base model, bitwise (target speed + checkpoints)")
ok(torch.equal(oan[1], ob[1]) and torch.equal(oan[2], ob[2]), "no coop kwargs == base model, bitwise")
ok(torch.allclose(oa1[1], ob[1], atol=1e-5) and torch.allclose(oa1[2], ob[2], atol=1e-5), "at initialisation rate 1 == base model (zero-initialised residual)")
with torch.no_grad(): _ = net_a(rgb=rgb, lidar_bev=lidar, target_point=tp, ego_vel=vel, command=cmd, coop_states=st, coop_mask=mk)   # refresh last_delta at rate 1
ok(net_a.v2x_adapter.last_delta is not None and float(net_a.v2x_adapter.last_delta.abs().max()) < 1e-6, "adapter residual is ~0 at initialisation")

# cache pipeline: hook the decoder output, apply adapter + frozen heads outside forward -> must reproduce forward
store = {}
h = net_a.join.register_forward_hook(lambda m, inp, out: store.update(memory=inp[1].detach(), joined=out.detach()))
with torch.no_grad(): oa1 = net_a(rgb=rgb, lidar_bev=lidar, target_point=tp, ego_vel=vel, command=cmd, coop_states=st, coop_mask=mk)
h.remove(); L = cfg_a.predict_checkpoint_len
print(f"decoder memory tokens S={store['memory'].shape[1]} (no coop tokens in memory), joined queries {tuple(store['joined'].shape)}")
ok(store["memory"].shape[1] == 65 or store["memory"].shape[1] > 0, "memory shape captured")
def heads(net, joined, tp, st, mk):
    q = net.v2x_adapter(joined, st, mk); ck = net.checkpoint_decoder(q[:, :L], tp); ts = net.target_speed_network(q[:, L]); return ts, ck
with torch.no_grad(): ts_h, ck_h = heads(net_a, store["joined"], tp, st, mk)
ok(torch.allclose(ts_h, oa1[1], atol=1e-5) and torch.allclose(ck_h, oa1[2], atol=1e-5), "heads(cached joined) == full forward (cache pipeline consistent)")
jh = store["joined"].half().float()
with torch.no_grad(): ts_h16, ck_h16 = heads(net_a, jh, tp, st, mk)
print(f"fp16 cache rounding: max |Δ logits| {float((ts_h16-oa1[1]).abs().max()):.2e}, max |Δ ckpt| {float((ck_h16-oa1[2]).abs().max()):.2e} m")
ok(float((ts_h16 - oa1[1]).abs().max()) < 1e-2 and float((ck_h16 - oa1[2]).abs().max()) < 1e-2, "fp16 feature cache rounding is negligible")

# training: only adapter params move; after training rate 1 differs from base while rate 0 stays bitwise base
for n_, p_ in net_a.named_parameters(): p_.requires_grad_(n_.startswith("v2x_adapter."))
params = [p for p in net_a.parameters() if p.requires_grad]; opt = torch.optim.AdamW(params, lr=1e-3)
ts_label = data["target_speed_twohot"].to(dev, dtype=torch.float32); ck_label = data["route"][:, :L].to(dev, dtype=torch.float32)
net_a.train()
for step in range(15):
    ts, ck = heads(net_a, store["joined"], tp, st, mk)
    loss = F.cross_entropy(ts, ts_label) + torch.abs(ck - ck_label).mean(); opt.zero_grad(); loss.backward()
    if step == 0:
        gn = [n_ for n_, p_ in net_a.named_parameters() if p_.grad is not None and float(p_.grad.abs().sum()) > 0]
        ok(len(gn) > 0 and all(n_.startswith("v2x_adapter.") for n_ in gn), f"gradients only on adapter params ({len(gn)} tensors with non-zero grad)")
        ok(any("out_proj" in n_ for n_ in gn) and any("mlp.2" in n_ for n_ in gn), "zero-initialised output projections receive gradient")
    opt.step()
net_a.eval()
with torch.no_grad():
    oa1t = net_a(rgb=rgb, lidar_bev=lidar, target_point=tp, ego_vel=vel, command=cmd, coop_states=st, coop_mask=mk)
    oa0t = net_a(rgb=rgb, lidar_bev=lidar, target_point=tp, ego_vel=vel, command=cmd, coop_states=st, coop_mask=torch.zeros_like(mk))
    obt = net_b(rgb=rgb, lidar_bev=lidar, target_point=tp, ego_vel=vel, command=cmd)
d1 = float((oa1t[1] - obt[1]).abs().mean())
ok(d1 > 1e-4, f"after 15 adapter steps rate 1 differs from base (mean |Δ logits| {d1:.4f})")
d0 = float((oa0t[1] - obt[1]).abs().max()); d0c = float((oa0t[2] - obt[2]).abs().max())
print(f"diag rate0 after training: max|Δ logits| {d0:.3e} max|Δ ckpt| {d0c:.3e}; nan logits {int(torch.isnan(oa0t[1]).sum())}; "
      f"base drift vs before: {float((obt[1] - ob[1]).abs().max()):.3e}; adapter last_delta at r0: max {float(net_a.v2x_adapter.last_delta.abs().max()):.3e}, nan {int(torch.isnan(net_a.v2x_adapter.last_delta).sum())}", flush=True)
with torch.no_grad():
    q0 = net_a.v2x_adapter(store["joined"], st, torch.zeros_like(mk)); ob2 = net_b(rgb=rgb, lidar_bev=lidar, target_point=tp, ego_vel=vel, command=cmd)
print(f"diag adapter module at r0: identity? {torch.equal(q0, store['joined'])} (max|Δ| {float((q0 - store['joined']).abs().max()):.3e}); "
      f"base forward repeat bitwise? {torch.equal(ob2[1], obt[1]) and torch.equal(ob2[2], obt[2])} (max|Δ| {float((ob2[1] - obt[1]).abs().max()):.3e})", flush=True)
ok(torch.equal(q0, store["joined"]), "after training the adapter module is an exact identity at rate 0")
ok(torch.allclose(oa0t[1], obt[1], atol=1e-4) and torch.allclose(oa0t[2], obt[2], atol=1e-4), "after training rate 0 == base model (full forward, atol 1e-4)")
ok(torch.allclose(obt[1], ob[1], atol=1e-4), "base model untouched by adapter training")
sd = net_a.state_dict(); net_c = modmod.LidarCenterNet(cfg_a); net_c.load_state_dict(sd, strict=True)
ok(True, "state_dict round-trips with strict=True (sensor_agent loads strict)")
js = jsonpickle.encode(cfg_a); cfg_r = jsonpickle.decode(js); ok(getattr(cfg_r, "use_v2x_adapter", 0) == 1 and cfg_r.v2x_adapter_layers == 2, "config.json round-trip keeps adapter fields")
# ---- deep mode: per-layer blocks inside the decoder ----
cfg_d = base_cfg(); cfg_d.use_v2x = 1; cfg_d.use_v2x_adapter = 1; cfg_d.v2x_adapter_deep = 1; cfg_d.v2x_adapter_calib = 0; cfg_d.v2x_adapter_ffn = 512
net_d = modmod.LidarCenterNet(cfg_d); res = net_d.load_state_dict(sd_base, strict=False); net_d.to(dev).eval()
ok(not res.unexpected_keys and all(k.startswith("v2x_adapter.") for k in res.missing_keys), f"deep: base weights load, missing = adapter only ({len(res.missing_keys)} tensors)")
n_deep = sum(p.numel() for n, p in net_d.named_parameters() if n.startswith("v2x_adapter.deep_layers")); print(f"deep blocks: {n_deep/1e6:.2f}M params over {len(net_d.v2x_adapter.deep_layers)} decoder layers")
with torch.no_grad():
    od0 = net_d(rgb=rgb, lidar_bev=lidar, target_point=tp, ego_vel=vel, command=cmd, coop_states=st, coop_mask=torch.zeros_like(mk))
    od1 = net_d(rgb=rgb, lidar_bev=lidar, target_point=tp, ego_vel=vel, command=cmd, coop_states=st, coop_mask=mk)
    odn = net_d(rgb=rgb, lidar_bev=lidar, target_point=tp, ego_vel=vel, command=cmd)
ok(torch.equal(od0[1], ob[1]) and torch.equal(od0[2], ob[2]), "deep: rate 0 == base bitwise (unrolled decoder + identity blocks == nn.TransformerDecoder)")
ok(torch.equal(odn[1], ob[1]) and torch.equal(odn[2], ob[2]), "deep: no coop kwargs == base bitwise")
ok(torch.allclose(od1[1], ob[1], atol=1e-5) and torch.allclose(od1[2], ob[2], atol=1e-5), "deep: rate 1 at initialisation == base (zero-init blocks)")
store_d = {}; h = net_d.join.register_forward_hook(lambda m, inp, out: store_d.update(memory=inp[1].detach(), joined=out.detach()))
with torch.no_grad(): _ = net_d(rgb=rgb, lidar_bev=lidar, target_point=tp, ego_vel=vel, command=cmd)
h.remove()
with torch.no_grad(): q_re = net_d.v2x_join(net_d.checkpoint_query.expand(rgb.shape[0], -1, -1), store_d["memory"], st, torch.zeros_like(mk))
ok(torch.equal(q_re, store_d["joined"]), "deep: v2x_join from the captured memory reproduces the decoder output bitwise (cache pipeline valid)")
for n_, p_ in net_d.named_parameters(): p_.requires_grad_(n_.startswith("v2x_adapter."))
opt_d = torch.optim.AdamW([p for p in net_d.parameters() if p.requires_grad], lr=1e-3); net_d.checkpoint_decoder.train()
for step in range(15):
    q = net_d.v2x_join(net_d.checkpoint_query.expand(rgb.shape[0], -1, -1), store_d["memory"], st, mk)
    ck = net_d.checkpoint_decoder(q[:, :L], tp); ts = net_d.target_speed_network(q[:, L])
    loss = F.cross_entropy(ts, ts_label) + torch.abs(ck - ck_label).mean(); opt_d.zero_grad(); loss.backward()
    if step == 0:
        gn = [n_ for n_, p_ in net_d.named_parameters() if p_.grad is not None and float(p_.grad.abs().sum()) > 0]
        ok(len(gn) > 0 and all(n_.startswith("v2x_adapter.deep_layers") for n_ in gn), f"deep: gradients only on the deep blocks ({len(gn)} tensors)")
        ok(len({n_.split('.')[2] for n_ in gn}) == len(net_d.v2x_adapter.deep_layers), "deep: every decoder layer's block receives gradient")
    opt_d.step()
net_d.eval()
with torch.no_grad():
    od1t = net_d(rgb=rgb, lidar_bev=lidar, target_point=tp, ego_vel=vel, command=cmd, coop_states=st, coop_mask=mk)
    od0t = net_d(rgb=rgb, lidar_bev=lidar, target_point=tp, ego_vel=vel, command=cmd, coop_states=st, coop_mask=torch.zeros_like(mk))
ok(float((od1t[1] - ob[1]).abs().mean()) > 1e-4, f"deep: after 15 steps rate 1 differs from base (mean |Δ logits| {float((od1t[1] - ob[1]).abs().mean()):.4f})")
ok(torch.equal(od0t[1], ob[1]) and torch.equal(od0t[2], ob[2]), "deep: after training rate 0 is still bitwise the base")
modmod.LidarCenterNet(cfg_d).load_state_dict(net_d.state_dict(), strict=True); ok(True, "deep: state_dict round-trips strict (sensor_agent loads strict)")
print("ADAPTER SMOKE PASSED")
