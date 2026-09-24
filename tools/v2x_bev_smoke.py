"""Structural checks for late fusion (use_v2x_bev): raster channels through the lidar stem, base checkpoint padded to zero.
  python tools/v2x_bev_smoke.py --root /work/gn21/n21001/carla_garage_smoke_root --base /work/gn21/n21001/carla_garage_runs/tfpp_base_000
"""
import argparse, os, sys, gzip, json
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "team_code"))
import jsonpickle, jsonpickle.ext.numpy as jsonpickle_numpy; jsonpickle_numpy.register_handlers()
import config as cfgmod, model as modmod, data as datamod, v2x_features as V

ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--base", required=True); a = ap.parse_args(); dev = "cuda"; torch.manual_seed(0)
def ok(c, m):
    print(("PASS " if c else "FAIL ") + m, flush=True)
    if not c: sys.exit("BEV SMOKE FAILED")
def base_cfg():
    cfg = cfgmod.GlobalConfig(); saved = jsonpickle.decode(open(os.path.join(a.base, "config.json")).read()); cfg.__dict__.update(saved.__dict__); return cfg
cfg_b = base_cfg(); cfg_f = base_cfg(); cfg_f.use_v2x = 0; cfg_f.use_v2x_bev = 1; cfg_f.v2x_bev_channels = 3; cfg_f.v2x_bev_width = 2.0
cfg_f.initialize(root_dir=[a.root], setting="all", use_v2x_bev=1); cfg_f.augment_percentage = 0.0; cfg_f.use_color_aug = 0; cfg_f.lidar_aug_prob = 0.0
ds = datamod.CARLA_Data(root=cfg_f.data_roots, config=cfg_f, estimate_class_distributions=False, estimate_sem_distribution=False, shared_dict=None, rank=0)
d = next(iter(torch.utils.data.DataLoader(torch.utils.data.Subset(ds, list(range(0, len(ds), max(1, len(ds) // 8)))[:8]), batch_size=8, shuffle=False, num_workers=0)))
ok("coop_states" in d and "coop_hidden" in d, "dataset emits coop states with use_v2x_bev=1 and use_v2x=0")
sd_base = torch.load(os.path.join(a.base, sorted(f for f in os.listdir(a.base) if f.startswith("model_"))[-1]), map_location="cpu")
net_b = modmod.LidarCenterNet(cfg_b); net_b.load_state_dict(sd_base, strict=True); net_b.to(dev).eval()
net_f = modmod.LidarCenterNet(cfg_f); sd_pad, padded = net_f.adapt_state_dict_for_bev_fusion(sd_base)
ok(len(padded) == 1 and "lidar_encoder" in padded[0], f"exactly one tensor padded (the lidar stem conv): {padded}")
res = net_f.load_state_dict(sd_pad, strict=True); net_f.to(dev).eval(); ok(True, "padded base checkpoint loads strict=True into the fusion model")
k = padded[0]; w = dict(net_f.named_parameters())[k]; print(f"stem weight {tuple(w.shape)}: histogram channel norm {float(w[:, :1].norm()):.3f}, raster channels norm {float(w[:, 1:].norm()):.3f}")
ok(float(w[:, 1:].norm()) == 0.0 and torch.equal(w[:, :1].cpu(), sd_base[k]), "raster channels start at exactly zero, histogram channel copied bitwise")

rgb = d["rgb"].to(dev, dtype=torch.float32); lidar = d["lidar"].to(dev, dtype=torch.float32); tp = d["target_point"].to(dev, dtype=torch.float32)
vel = d["speed"].to(dev, dtype=torch.float32).unsqueeze(1); cmd = d["command"].to(dev, dtype=torch.float32)
st = d["coop_states"].to(dev, dtype=torch.float32); mk = d["coop_mask"].to(dev, dtype=torch.float32); hid = d["coop_hidden"].to(dev)
raster = V.rasterize_states(st, mk, cfg_f.min_x, cfg_f.max_x, cfg_f.min_y, cfg_f.max_y, cfg_f.pixels_per_meter, cfg_f.v2x_bev_width)
ok(raster.shape == (8, 3, 256, 256) and float(raster[:, 0].sum()) > 0, f"raster built: {int(mk.sum())} vehicles -> {int(raster[:, 0].sum())} occupied px")
# alignment: visible vehicles (lidar hits in the recorded box) must have lidar histogram mass inside their raster footprint
vis = (mk > 0.5) & (hid < 0.5); per = []
for b in range(8):
    for j in torch.nonzero(vis[b]).flatten().tolist():
        one = V.rasterize_states(st[b:b+1, j:j+1], mk[b:b+1, j:j+1], cfg_f.min_x, cfg_f.max_x, cfg_f.min_y, cfg_f.max_y, cfg_f.pixels_per_meter, cfg_f.v2x_bev_width)[0, 0]
        if one.sum() > 0: per.append(float((lidar[b, 0] * one).sum() > 0))
print(f"visible vehicles with lidar mass inside their raster footprint: {int(sum(per))}/{len(per)}")
ok(len(per) >= 10 and sum(per) / len(per) > 0.8, "raster footprints align with the lidar histogram (>80% of visible vehicles have hits inside)")
lid_f = torch.cat((lidar, raster), dim=1)
with torch.no_grad():
    ob = net_b(rgb=rgb, lidar_bev=lidar, target_point=tp, ego_vel=vel, command=cmd)
    of = net_f(rgb=rgb, lidar_bev=lid_f, target_point=tp, ego_vel=vel, command=cmd)
    of0 = net_f(rgb=rgb, lidar_bev=torch.cat((lidar, torch.zeros_like(raster)), 1), target_point=tp, ego_vel=vel, command=cmd)
ok(torch.equal(of[1], ob[1]) and torch.equal(of[2], ob[2]), "at init the fusion model == base bitwise even with a non-empty raster (zero stem weights)")
ok(torch.equal(of0[1], ob[1]), "empty raster == base bitwise")
# one training step: gradient must reach the raster channels of the stem and the output must start depending on the raster
net_f.train(); opt = torch.optim.AdamW(net_f.parameters(), lr=1e-4)
ts_label = d["target_speed_twohot"].to(dev, dtype=torch.float32); ck_label = d["route"][:, :cfg_f.predict_checkpoint_len].to(dev, dtype=torch.float32)
for step in range(10):
    o = net_f(rgb=rgb, lidar_bev=lid_f, target_point=tp, ego_vel=vel, command=cmd)
    loss = F.cross_entropy(o[1], ts_label) + torch.abs(o[2] - ck_label).mean(); opt.zero_grad(); loss.backward()
    if step == 0: ok(w.grad is not None and float(w.grad[:, 1:].abs().sum()) > 0, f"gradient reaches the raster stem channels (|g| {float(w.grad[:, 1:].abs().sum()):.2e})")
    opt.step()
net_f.eval()
with torch.no_grad():
    o1 = net_f(rgb=rgb, lidar_bev=lid_f, target_point=tp, ego_vel=vel, command=cmd); o0 = net_f(rgb=rgb, lidar_bev=torch.cat((lidar, torch.zeros_like(raster)), 1), target_point=tp, ego_vel=vel, command=cmd)
ok(float((o1[1] - o0[1]).abs().mean()) > 1e-5, f"after 10 steps the output depends on the raster (mean |Δ logits| with vs without raster {float((o1[1]-o0[1]).abs().mean()):.2e})")
modmod.LidarCenterNet(cfg_f).load_state_dict(net_f.state_dict(), strict=True); ok(True, "state_dict round-trips strict")
js = jsonpickle.encode(cfg_f); cr = jsonpickle.decode(js); ok(cr.use_v2x_bev == 1 and cr.v2x_bev_channels == 3 and cr.use_v2x == 0, "config.json round-trip keeps use_v2x_bev / channels / use_v2x=0")
print("BEV SMOKE PASSED")
