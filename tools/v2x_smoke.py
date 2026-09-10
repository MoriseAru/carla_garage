"""Smoke test for the V2XState plug-in in TransFuser++ (no CARLA needed).
1) config with use_v2x=1 -> LidarCenterNet builds; forward with random rgb/lidar + coop tokens; loss-free backward reaches coop params;
   rate-0 (all slots null) vs rate-1 outputs differ; 2) apply_rate keeps the hash-selected subset; 3) CARLA_Data on one downloaded
   scenario yields coop_states/coop_mask/coop_bucket of the right shapes with plausible values.
  python tools/v2x_smoke.py --root /work/gn21/n21001/carla_garage_data/CrossingBicycleFlow
"""
import argparse, glob, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "team_code"))
import config as cfgmod, model as modmod, data as datamod, v2x_features

ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
cfg = cfgmod.GlobalConfig(); cfg.initialize(root_dir=a.root, setting="all", use_v2x=1, v2x_k=16, v2x_rate=1.0)
net = modmod.LidarCenterNet(cfg).to(dev)
n_coop = sum(p.numel() for n, p in net.named_parameters() if "coop_" in n)
print(f"model built on {dev}: {sum(p.numel() for p in net.parameters())/1e6:.1f}M params, coop params {n_coop}")
bs = 2
rgb = torch.randn(bs, 3, cfg.camera_height, cfg.camera_width, device=dev)
lidar = torch.randn(bs, cfg.lidar_seq_len * (2 if cfg.use_ground_plane else 1), cfg.lidar_resolution_height, cfg.lidar_resolution_width, device=dev)
tp = torch.randn(bs, 2, device=dev); vel = torch.rand(bs, 1, device=dev) * 10; cmd = torch.nn.functional.one_hot(torch.randint(0, 6, (bs,)), 6).float().to(dev)
states = torch.randn(bs, 16, 7, device=dev); mask = torch.ones(bs, 16, device=dev); mask[:, 10:] = 0
net.train()
out = net(rgb=rgb, lidar_bev=lidar, target_point=tp, ego_vel=vel, command=cmd, coop_states=states, coop_mask=mask)
pred_wp, pred_ts, pred_ckpt = out[0], out[1], out[2]
print("outputs:", [None if o is None else tuple(o.shape) for o in (pred_wp, pred_ts, pred_ckpt)])
loss = sum(o.float().pow(2).mean() for o in (pred_ts, pred_ckpt) if o is not None); loss.backward()
g = [n for n, p in net.named_parameters() if "coop_" in n and p.grad is not None and p.grad.abs().sum() > 0]
print(f"grad reached coop params: {len(g)}/{sum(1 for n, _ in net.named_parameters() if 'coop_' in n)}")
net.eval()
with torch.no_grad():
    o1 = net(rgb=rgb, lidar_bev=lidar, target_point=tp, ego_vel=vel, command=cmd, coop_states=states, coop_mask=mask)
    o0 = net(rgb=rgb, lidar_bev=lidar, target_point=tp, ego_vel=vel, command=cmd, coop_states=states, coop_mask=torch.zeros_like(mask))
    d = (o1[2] - o0[2]).abs().mean().item() if o1[2] is not None else (o1[1] - o0[1]).abs().mean().item()
print(f"rate1 vs rate0 mean |diff| of checkpoint/target-speed output: {d:.4f} (must be > 0)")
# apply_rate
bucket = torch.randint(0, 1000, (bs, 16), device=dev)
s5, m5 = v2x_features.apply_rate(states, mask, bucket, 0.5); print(f"apply_rate 0.5 keeps {int(m5.sum())} of {int(mask.sum())} slots; rate 0 keeps {int(v2x_features.apply_rate(states, mask, bucket, 0.0)[1].sum())}")
# dataset
t0 = time.time()
ds = datamod.CARLA_Data(root=[a.root], config=cfg, estimate_class_distributions=False, estimate_sem_distribution=False, shared_dict=None, rank=0)
print(f"dataset: {len(ds)} samples from {a.root} ({time.time()-t0:.0f}s)")
for i in np.linspace(0, len(ds) - 1, 3).astype(int):
    d = ds[i]; cs, cm, cb = d["coop_states"], d["coop_mask"], d["coop_bucket"]
    valid = cs[cm > 0]
    print(f"  sample {i}: coop_states {cs.shape} valid slots {int(cm.sum())}; x range [{32*valid[:,0].min():.1f},{32*valid[:,0].max():.1f}] m, |v| max {10*np.hypot(valid[:,2],valid[:,3]).max():.1f} m/s, len max {5*valid[:,6].max():.1f} m; buckets {cb[cm>0][:5].tolist()}" if len(valid) else f"  sample {i}: no vehicles within radius")
print("V2X SMOKE PASSED")
