"""Offline diagnosis of cooperation dependence: how do the V2X model's predictions degrade as the penetration rate drops,
and is the rate-0 collapse an out-of-distribution effect of 16 null tokens or a loss of information?
  python tools/v2x_offline_rate_sweep.py --root /work/gn21/n21001/carla_garage_data_root --v2x <run_dir> --base <run_dir> [--n 3000]
Conditions on the same frames: base model; v2x at rate 1 / .75 / .5 / .25 / 0 (masked slots -> null token, as in closed loop);
v2x at rate 0 with the coop tokens REMOVED from the decoder memory ("drop") instead of null-filled.
Metrics vs expert labels: target-speed class accuracy / CE, checkpoint L1 (m), plus the coop-slot fill statistics of the data.
"""
import argparse, os, sys, json
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "team_code"))
import jsonpickle, jsonpickle.ext.numpy as jsonpickle_numpy; jsonpickle_numpy.register_handlers()
import config as cfgmod, model as modmod, data as datamod, v2x_features

ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--v2x", required=True); ap.add_argument("--base", required=True)
ap.add_argument("--n", type=int, default=3000); ap.add_argument("--bs", type=int, default=16); a = ap.parse_args()
dev = "cuda"


def load(run_dir):
    cfg = cfgmod.GlobalConfig(); saved = jsonpickle.decode(open(os.path.join(run_dir, "config.json")).read()); cfg.__dict__.update(saved.__dict__)
    net = modmod.LidarCenterNet(cfg); ck = sorted(f for f in os.listdir(run_dir) if f.startswith("model_") and f.endswith(".pth"))[-1]
    net.load_state_dict(torch.load(os.path.join(run_dir, ck), map_location="cpu"), strict=False); net.to(dev).eval(); return cfg, net


cfg_v, net_v = load(a.v2x); cfg_b, net_b = load(a.base)
cfg_v.initialize(root_dir=[a.root], setting="all", use_v2x=1)   # dataset with coop states (augmentation kept: identical batch for every condition)
ds = datamod.CARLA_Data(root=cfg_v.data_roots, config=cfg_v, estimate_class_distributions=False, estimate_sem_distribution=False, shared_dict=None, rank=0)
g = torch.Generator().manual_seed(0); idx = torch.randperm(len(ds), generator=g)[: a.n].tolist()
sub = torch.utils.data.Subset(ds, idx); dl = torch.utils.data.DataLoader(sub, batch_size=a.bs, shuffle=False, num_workers=16)
print(f"frames {len(sub)} of {len(ds)}; K={cfg_v.v2x_k}", flush=True)

conds = ["base", "v2x_r1.0", "v2x_r0.75", "v2x_r0.5", "v2x_r0.25", "v2x_r0", "v2x_r0_drop"]
acc = {c: 0 for c in conds}; ce = {c: 0.0 for c in conds}; l1 = {c: 0.0 for c in conds}; n = 0; fill = []; any_null = 0
orig_coop_tokens = net_v.coop_tokens
def drop_tokens(states, mask, bs): return torch.zeros(bs, 0, cfg_v.gru_input_size, device=dev)

with torch.no_grad():
    for data in dl:
        rgb = data["rgb"].to(dev, dtype=torch.float32); lidar = data["lidar"].to(dev, dtype=torch.float32)
        tp = data["target_point"].to(dev, dtype=torch.float32); tpn = data["target_point_next"].to(dev, dtype=torch.float32) if cfg_v.two_tp_input else None
        vel = data["speed"].to(dev, dtype=torch.float32).unsqueeze(1); cmd = data["command"].to(dev, dtype=torch.float32)
        ts_label = data["target_speed_twohot"].to(dev, dtype=torch.float32).argmax(1)   # class index of the expert target speed
        ckpt_label = data["route"][:, : cfg_v.predict_checkpoint_len].to(dev, dtype=torch.float32)
        if n == 0: print("label check: ts classes", int(ts_label.min()), "-", int(ts_label.max()), "of", len(cfg_v.target_speeds), "; ckpt", tuple(ckpt_label.shape), "; cmd", tuple(cmd.shape), flush=True)
        st = data["coop_states"].to(dev, dtype=torch.float32); mk = data["coop_mask"].to(dev, dtype=torch.float32); bk = data["coop_bucket"].to(dev)
        fill += mk.sum(1).tolist(); any_null += int((mk.sum(1) < cfg_v.v2x_k).sum()); bs = rgb.shape[0]
        for c in conds:
            if c == "base":
                out = net_b(rgb=rgb, lidar_bev=lidar, target_point=tp, ego_vel=vel, command=cmd, target_point_next=tpn)
            else:
                if c == "v2x_r0_drop":
                    net_v.coop_tokens = drop_tokens; s, m = st, torch.zeros_like(mk)
                else:
                    net_v.coop_tokens = orig_coop_tokens; s, m = v2x_features.apply_rate(st, mk, bk, float(c.split("_r")[1]))
                out = net_v(rgb=rgb, lidar_bev=lidar, target_point=tp, ego_vel=vel, command=cmd, target_point_next=tpn, coop_states=s, coop_mask=m)
            pred_ts, pred_ckpt = out[1], out[2]
            acc[c] += int((pred_ts.argmax(1) == ts_label).sum()); ce[c] += float(F.cross_entropy(pred_ts, ts_label, reduction="sum"))
            l1[c] += float((pred_ckpt - ckpt_label).abs().mean(dim=(1, 2)).sum())
        n += bs
net_v.coop_tokens = orig_coop_tokens
fill = np.array(fill)
print(f"\ncoop slots: mean filled {fill.mean():.1f}/{cfg_v.v2x_k}; frames with >=1 null slot {100*any_null/n:.0f}%; frames with 0 vehicles {100*np.mean(fill==0):.1f}%; frames with <=4 vehicles {100*np.mean(fill<=4):.0f}%")
print(f"\n{'condition':14s} {'ts acc%':>8s} {'ts CE':>7s} {'ckpt L1 m':>10s}")
for c in conds: print(f"{c:14s} {100*acc[c]/n:8.1f} {ce[c]/n:7.3f} {l1[c]/n:10.3f}")
json.dump(dict(n=n, acc={c: acc[c]/n for c in conds}, ce={c: ce[c]/n for c in conds}, l1={c: l1[c]/n for c in conds}, fill_mean=float(fill.mean()), frac_any_null=any_null/n),
          open("/work/gn21/n21001/V2XState_Real/tmp/v2x_offline_rate_sweep.json", "w"), indent=1)
print("RATE_SWEEP_DONE")
