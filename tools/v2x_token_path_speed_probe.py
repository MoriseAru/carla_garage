"""Offline probe (HANDOVER 9943, LOCAL request): on the training distribution, do the GT state tokens of the token model
improve the target-speed prediction only, or also the path (checkpoint) prediction?
  python tools/v2x_token_path_speed_probe.py --root /work/gn21/n21001/carla_garage_data_root \
      --v2x <tfpp_v2x_000 dir> --base <tfpp_base_000 dir> --n 20000 --shard 0 --nshard 4
Frames: a seeded random subset (seed 0) of the full TF++ dataset, un-augmented (augment_percentage 0, no colour / lidar aug),
identical for every condition and both models; --shard/--nshard split it into disjoint slices (merge with --merge).
Conditions (token model only): a = GT tokens as stored; b = all tokens cleared (mask 0, null-filled as in closed loop);
c = velocity dims (vx/10, vy/10) zeroed, everything else kept; d = tokens (states + mask) rolled by one sample in the batch;
e = the same frame's 16 slots (states + mask together) in a random order (same vehicles, slot identity changed; HANDOVER 10169).
base = the sensor-only model (no tokens).
Per frame and condition: checkpoint L1 (mean over 10 checkpoints and x/y, m) plus its lateral (y) part, target-speed argmax
correct, expected speed (sum_k P_k v_k) and controller speed (0 if P(stop) > brake_uncertainty_threshold); expert speed =
two-hot . target_speeds. Label side: lateral coordinate of the 10th route checkpoint, and whether a GT vehicle (token
states, K=16 nearest within 64 m) within 32 m moves faster than 1 m/s.
Read-only: loads the latest model_*.pth of each run dir (model_0030), writes only --out.
"""
import argparse, glob, json, math, os, sys
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "team_code"))
import jsonpickle, jsonpickle.ext.numpy as jsonpickle_numpy; jsonpickle_numpy.register_handlers()

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="/work/gn21/n21001/carla_garage_data_root")
ap.add_argument("--v2x", default="/work/gn21/n21001/carla_garage_runs/tfpp_v2x_000")
ap.add_argument("--base", default="/work/gn21/n21001/carla_garage_runs/tfpp_base_000")
ap.add_argument("--n", type=int, default=20000); ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--offset", type=int, default=0, help="frames perm[offset : offset + n] of the seed-0 permutation (0 = the earlier probes)")
ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshard", type=int, default=1)
ap.add_argument("--out", default="/work/gn21/n21001/V2XState_Real/tmp/v2x_token_probe")
ap.add_argument("--merge", type=int, default=0, help="1: only merge <out>_shard*.npz and print the tables")
ap.add_argument("--lon", type=int, default=0, help="1: install the longitudinal-only decoder masks on the --v2x model")
ap.add_argument("--heldout", type=int, default=0, help="1: --root holds only routes the training filter drops (expert score < 100, "
                "data.py:101-117); the filter is bypassed while the dataset index is built so those routes are loaded")
a = ap.parse_args()
CONDS = ["base", "a_gt", "b_clear", "c_novel", "d_shuffle", "e_perm"]
TOKC = ("b_clear", "c_novel", "d_shuffle", "e_perm")


def report(D):
    n = len(D["lat10"]); print(f"\nframes {n}")
    v_exp = D["v_exp"]
    print(f"{'condition':10s} {'ckpt L1 m':>9s} {'lat L1 m':>8s} {'ts acc%':>8s} {'|E[v]-v*|':>10s} {'|v_ctrl-v*|':>11s}")
    res = {}
    for c in CONDS:
        r = dict(ckpt_l1=float(D[c + "_l1"].mean()), lat_l1=float(D[c + "_latl1"].mean()), ts_acc=float(D[c + "_ok"].mean()),
                 speed_mae=float(np.abs(D[c + "_ev"] - v_exp).mean()), ctrl_speed_mae=float(np.abs(D[c + "_vc"] - v_exp).mean()))
        res[c] = r
        print(f"{c:10s} {r['ckpt_l1']:9.3f} {r['lat_l1']:8.3f} {100*r['ts_acc']:8.2f} {r['speed_mae']:10.3f} {r['ctrl_speed_mae']:11.3f}")
    print("\noutput change vs a_gt (token model): mean |d ckpt| m, frames with max lateral shift > 0.5 m, mean |d E[v]| m/s, frames > 1 m/s")
    for c in TOKC:
        r = res[c]; r["d_ckpt"] = float(D[c + "_dck"].mean()); r["frac_lat_gt05"] = float((D[c + "_dlat"] > 0.5).mean())
        r["d_v"] = float(D[c + "_dv"].mean()); r["frac_dv_gt1"] = float((D[c + "_dv"] > 1.0).mean())
        print(f"  {c:10s} {r['d_ckpt']:.3f}  {100*r['frac_lat_gt05']:5.2f}%  {r['d_v']:.3f}  {100*r['frac_dv_gt1']:5.2f}%")
    mv = D["moving32"].astype(bool); lab = {}
    for name, sel in (("moving_vehicle_within_32m", mv), ("none", ~mv)):
        lab[name] = dict(n=int(sel.sum()), lat10_std=float(D["lat10"][sel].std()) if sel.any() else None,
                         lat10_abs_mean=float(np.abs(D["lat10"][sel]).mean()) if sel.any() else None)
        print(f"label: {name:26s} n={lab[name]['n']:6d}  std(lat of 10th checkpoint) {lab[name]['lat10_std']:.3f} m  mean|lat| {lab[name]['lat10_abs_mean']:.3f} m")
    return dict(n=n, conds=res, label=lab)


if a.merge:
    fs = sorted(glob.glob(a.out + "_shard*.npz")); print("merging", fs)
    parts = [dict(np.load(f)) for f in fs]; D = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    json.dump(report(D) | {"shards": fs}, open(a.out + ".json", "w"), indent=1); print("PROBE_MERGED"); sys.exit(0)

import config as cfgmod, model as modmod, data as datamod  # noqa: E402
dev = "cuda"


def load(run_dir):
    cfg = cfgmod.GlobalConfig(); saved = jsonpickle.decode(open(os.path.join(run_dir, "config.json")).read()); cfg.__dict__.update(saved.__dict__)
    net = modmod.LidarCenterNet(cfg); ck = sorted(f for f in os.listdir(run_dir) if f.startswith("model_") and f.endswith(".pth"))[-1]
    missing = net.load_state_dict(torch.load(os.path.join(run_dir, ck), map_location="cpu"), strict=False)
    print(run_dir, ck, "missing", len(missing.missing_keys), "unexpected", len(missing.unexpected_keys), flush=True)
    net.to(dev).eval(); return cfg, net


cfg_v, net_v = load(a.v2x); cfg_b, net_b = load(a.base)
if a.lon:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import v2x_lon_mask; v2x_lon_mask.install(net_v)
    print("lon-only decoder masks installed on", a.v2x, flush=True)
cfg_v.initialize(root_dir=[a.root], setting="all", use_v2x=1, use_v2x_bev=0)
cfg_v.augment_percentage = 0.0; cfg_v.use_color_aug = 0; cfg_v.lidar_aug_prob = 0.0
_ujson = datamod.ujson
if a.heldout:
    class _PassFilter:   # results.json reports a perfect expert so data.py keeps the route; every other json is untouched
        @staticmethod
        def load(f):
            d = _ujson.load(f)
            if isinstance(d, dict) and "scores" in d and "infractions" in d:
                d = dict(d, scores=dict(d["scores"], score_composed=100.0), status="Completed")
            return d
    datamod.ujson = _PassFilter
ds = datamod.CARLA_Data(root=cfg_v.data_roots, config=cfg_v, estimate_class_distributions=False, estimate_sem_distribution=False, shared_dict=None, rank=0)
datamod.ujson = _ujson
g = torch.Generator().manual_seed(0)
idx = torch.randperm(len(ds), generator=g)[a.offset: a.offset + a.n].tolist()[a.shard:: a.nshard]
dl = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, idx), batch_size=a.bs, shuffle=False, num_workers=16, drop_last=True)
print(f"dataset {len(ds)} frames; shard {a.shard}/{a.nshard}: {len(idx)} frames (drop_last batch {a.bs})", flush=True)
gp = torch.Generator().manual_seed(1)   # e_perm: same slot permutations for every model
spd_t = torch.tensor(cfg_v.target_speeds, dtype=torch.float32, device=dev); thr = cfg_v.brake_uncertainty_threshold
K = {k: [] for k in ["v_exp", "lat10", "moving32"] + [c + s for c in CONDS for s in ("_l1", "_latl1", "_ok", "_ev", "_vc")]
     + [c + s for c in TOKC for s in ("_dck", "_dlat", "_dv")]}
out_path = f"{a.out}_shard{a.shard}.npz"


def dump():
    np.savez_compressed(out_path, **{k: np.asarray(v) for k, v in K.items()})


with torch.no_grad():
    for bi, data in enumerate(dl):
        rgb = data["rgb"].to(dev, dtype=torch.float32); lidar = data["lidar"].to(dev, dtype=torch.float32)
        tp = data["target_point"].to(dev, dtype=torch.float32)
        vel = data["speed"].to(dev, dtype=torch.float32).unsqueeze(1); cmd = data["command"].to(dev, dtype=torch.float32)
        tw = data["target_speed_twohot"].to(dev, dtype=torch.float32); ts_label = tw.argmax(1); v_exp = tw @ spd_t
        ck = data["route"][:, : cfg_v.predict_checkpoint_len].to(dev, dtype=torch.float32)
        st = data["coop_states"].to(dev, dtype=torch.float32); mk = data["coop_mask"].to(dev, dtype=torch.float32)
        if bi == 0:
            print("label check: ckpt", tuple(ck.shape), "ts classes", int(ts_label.min()), "-", int(ts_label.max()), "tokens/frame", float(mk.sum(1).mean()), flush=True)
        K["v_exp"] += v_exp.tolist(); K["lat10"] += ck[:, -1, 1].tolist()
        dist = torch.linalg.norm(st[..., :2] * 32.0, dim=-1); spd = torch.linalg.norm(st[..., 2:4] * 10.0, dim=-1)
        K["moving32"] += ((mk > 0.5) & (dist < 32.0) & (spd > 1.0)).any(1).tolist()
        sv = st.clone(); sv[..., 2:4] = 0.0
        perm = torch.stack([torch.randperm(st.shape[1], generator=gp) for _ in range(st.shape[0])]).to(dev)   # per-frame slot order
        sp = torch.gather(st, 1, perm[..., None].expand_as(st)); mp = torch.gather(mk, 1, perm)
        inputs = {"a_gt": (st, mk), "b_clear": (torch.zeros_like(st), torch.zeros_like(mk)), "c_novel": (sv, mk),
                  "d_shuffle": (torch.roll(st, 1, 0), torch.roll(mk, 1, 0)), "e_perm": (sp, mp)}
        outs = {}
        for c in CONDS:
            if c == "base":
                o = net_b(rgb=rgb, lidar_bev=lidar, target_point=tp, ego_vel=vel, command=cmd, target_point_next=None)
            else:
                s, m = inputs[c]
                o = net_v(rgb=rgb, lidar_bev=lidar, target_point=tp, ego_vel=vel, command=cmd, target_point_next=None, coop_states=s, coop_mask=m)
            pts, pck = o[1].float(), o[2].float(); P = F.softmax(pts, 1); ev = P @ spd_t
            vc = torch.where(P[:, 0] > thr, torch.zeros_like(ev), ev); outs[c] = (pck, ev)
            K[c + "_l1"] += (pck - ck).abs().mean(dim=(1, 2)).tolist(); K[c + "_latl1"] += (pck[..., 1] - ck[..., 1]).abs().mean(1).tolist()
            K[c + "_ok"] += (pts.argmax(1) == ts_label).tolist(); K[c + "_ev"] += ev.tolist(); K[c + "_vc"] += vc.tolist()
        for c in TOKC:
            d = outs[c][0] - outs["a_gt"][0]
            K[c + "_dck"] += d.abs().mean(dim=(1, 2)).tolist(); K[c + "_dlat"] += d[..., 1].abs().amax(1).tolist()
            K[c + "_dv"] += (outs[c][1] - outs["a_gt"][1]).abs().tolist()
        if bi % 50 == 0:
            dump(); print(f"batch {bi} frames {len(K['v_exp'])}", flush=True)
dump()
report({k: np.asarray(v) for k, v in K.items()})
print("PROBE_SHARD_DONE", out_path)
