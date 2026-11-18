"""Read-only (LOCAL HANDOVER 10475): does tfpp_v2x_000 attend to / depend on the expert's decisive object?
Frames: the 20,000 training frames of 9973 (seed-0 randperm, setting 'all', augmentation off), sharded as the token probe.
Decisive object = measurement speed_reduced_by_obj_id of the current frame. A frame counts when that id is a car box that
occupies one of the 16 token slots (slot order rebuilt exactly as v2x_features.coop_states_from_boxes: car boxes within
v2x_radius, sorted by distance; checked against the loaded coop_states).
 1. attention: speed query (row predict_checkpoint_len) cross-attention weights of each decoder layer (heads averaged) on the
    16 token columns; top-1 / top-3 among valid slots for the last layer and the 6-layer mean; share of the token mass;
    chance = mean 1 / n_valid (top-3: mean min(3, n)/n).
 2. influence: replace one valid slot at a time by coop_null in the decoder memory (backbone memory reused), recompute
    E[v] = softmax(target speed) . speeds; top-1 = decisive slot gives the largest |dE[v]|.
 3. decisive object: speed (box 'speed') > 0.5 m/s or not; ahead within 12 m in the ego lane (0 < x <= 12, |y| <= 1.75 m
    and |y| <= 2.5 m as lane proxies; boxes carry no lane id).
"""
import argparse, glob, gzip, json, math, os, sys
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "team_code"))
import jsonpickle, jsonpickle.ext.numpy as jsonpickle_numpy; jsonpickle_numpy.register_handlers()

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="/work/gn21/n21001/carla_garage_data_root")
ap.add_argument("--v2x", default="/work/gn21/n21001/carla_garage_runs/tfpp_v2x_000")
ap.add_argument("--n", type=int, default=20000); ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshard", type=int, default=1)
ap.add_argument("--out", default="/work/gn21/n21001/V2XState_Real/tmp/v2x_decisive")
ap.add_argument("--merge", type=int, default=0)
ap.add_argument("--lon", type=int, default=0, help="1: install the longitudinal-only decoder masks (tools/v2x_lon_mask.py)")
ap.add_argument("--offset", type=int, default=0, help="frames perm[offset : offset + n] of the seed-0 permutation "
                "(24000: frames the 1500-step x 16 fine-tunes did not use)")
a = ap.parse_args()


def report(D):
    groups = {}
    if len(D["slot"]):
        s_, mv = D["slot"], D["dspeed"] > 0.5
        for g, m in [("nearest", s_ == 0), ("not_nearest", s_ >= 1), ("slot_ge3", s_ >= 3), ("moving", mv), ("static", ~mv)]:
            groups[g] = dict(n=int(m.sum()), chance_top1=float((1.0 / D["nvalid"][m]).mean()), att_last_top1=float((D["rank_last"][m] == 0).mean()),
                             att_mean_top1=float((D["rank_mean"][m] == 0).mean()), infl_top1=float((D["rank_infl"][m] == 0).mean()))
    r = {"groups": groups,"frames_total": int(D["n_frames"].sum()), "frames_with_decisive_id": int(D["n_dec"].sum()),
         "frames_decisive_in_token_slots": int(len(D["slot"]))}
    n = len(D["slot"])
    if n == 0:
        return r
    nv = D["nvalid"]
    r["mean_valid_tokens"] = float(nv.mean())
    r["chance_top1"] = float((1.0 / nv).mean()); r["chance_top3"] = float((np.minimum(3, nv) / nv).mean())
    for k in ("last", "mean"):
        r[f"att_{k}_top1"] = float(D[f"rank_{k}"].__eq__(0).mean()); r[f"att_{k}_top3"] = float((D[f"rank_{k}"] < 3).mean())
        r[f"att_{k}_share"] = float(D[f"share_{k}"].mean())
    r["att_per_layer_top1"] = [float((D["rank_layers"][:, l] == 0).mean()) for l in range(D["rank_layers"].shape[1])]
    r["infl_top1"] = float((D["rank_infl"] == 0).mean()); r["infl_top3"] = float((D["rank_infl"] < 3).mean())
    r["infl_decisive_mean_abs_dEv"] = float(D["dv_dec"].mean()); r["infl_max_mean_abs_dEv"] = float(D["dv_max"].mean())
    r["infl_decisive_frac_dEv_gt1"] = float((D["dv_dec"] > 1.0).mean())
    r["infl_top1_when_max_dEv_gt_0.1"] = float((D["rank_infl"][D["dv_max"] > 0.1] == 0).mean()) if (D["dv_max"] > 0.1).any() else None
    r["frames_max_dEv_gt_0.1"] = int((D["dv_max"] > 0.1).sum())
    r["att_infl_same_slot_last"] = float(D["att_infl_same"].mean())
    r["decisive_moving_gt_0.5"] = float((D["dspeed"] > 0.5).mean())
    ahead = (D["dx"] > 0) & (D["dx"] <= 12.0)
    r["decisive_ahead12_lane1.75"] = float((ahead & (np.abs(D["dy"]) <= 1.75)).mean())
    r["decisive_ahead12_lane2.5"] = float((ahead & (np.abs(D["dy"]) <= 2.5)).mean())
    r["decisive_distance_quantiles_5_25_50_75_95"] = np.percentile(np.hypot(D["dx"], D["dy"]), [5, 25, 50, 75, 95]).round(2).tolist()
    r["decisive_slot_quantiles_5_25_50_75_95"] = np.percentile(D["slot"], [5, 25, 50, 75, 95]).tolist()
    r["slot_check_max_abs_err_m"] = float(D["slot_err"].max())
    return r


if a.merge:
    fs = sorted(glob.glob(a.out + "_shard*.npz")); parts = [dict(np.load(f)) for f in fs]
    D = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    r = report(D) | {"shards": fs}; json.dump(r, open(a.out + ".json", "w"), indent=1); print(json.dumps(r, indent=1)); sys.exit(0)

import config as cfgmod, model as modmod, data as datamod  # noqa: E402
dev = "cuda"
cfg = cfgmod.GlobalConfig(); saved = jsonpickle.decode(open(os.path.join(a.v2x, "config.json")).read()); cfg.__dict__.update(saved.__dict__)
net = modmod.LidarCenterNet(cfg)
net.load_state_dict(torch.load(os.path.join(a.v2x, "model_0030.pth"), map_location="cpu"), strict=True)
net.to(dev).eval()
if a.lon:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import v2x_lon_mask; v2x_lon_mask.install(net)
    print("lon-only decoder masks installed", flush=True)
P, K, R = int(cfg.predict_checkpoint_len), int(cfg.v2x_k), float(cfg.v2x_radius)
assert not cfg.input_path_to_target_speed_network

# capture: decoder memory/query (pre-hook) and per-layer cross-attention weights
cap = {"on": False, "att": [], "q": None, "mem": None}


def pre(mod, args, kwargs):
    if cap["on"]:
        cap["q"], cap["mem"] = args[0].detach(), args[1].detach()


net.join.register_forward_pre_hook(pre, with_kwargs=True)
for layer in net.join.layers:
    mha = layer.multihead_attn; orig = mha.forward

    def f(*args, _o=orig, **kw):
        kw["need_weights"] = True; kw["average_attn_weights"] = True
        out, w = _o(*args, **kw)
        if cap["on"]:
            cap["att"].append(w[:, P].detach())   # (bs, S): speed query row
        return out, w
    mha.forward = f

cfg.initialize(root_dir=[a.root], setting="all", use_v2x=1, use_v2x_bev=0)
cfg.augment_percentage = 0.0; cfg.use_color_aug = 0; cfg.lidar_aug_prob = 0.0
ds = datamod.CARLA_Data(root=cfg.data_roots, config=cfg, estimate_class_distributions=False, estimate_sem_distribution=False,
                        shared_dict=None, rank=0)
idx = torch.randperm(len(ds), generator=torch.Generator().manual_seed(0))[a.offset: a.offset + a.n].tolist()[a.shard:: a.nshard]
dl = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, idx), batch_size=a.bs, shuffle=False, num_workers=16, drop_last=True)
cur = cfg.seq_len - 1
spd_t = torch.tensor(cfg.target_speeds, dtype=torch.float32, device=dev)
print(f"dataset {len(ds)}; shard {a.shard}/{a.nshard}: {len(idx)} frames", flush=True)


def rd(p):
    with gzip.open(p, "rt", encoding="utf-8") as fh:
        return json.load(fh)


Kk = ["slot", "nvalid", "rank_last", "rank_mean", "share_last", "share_mean", "rank_infl", "dv_dec", "dv_max", "att_infl_same",
      "dspeed", "dx", "dy", "slot_err"]
O = {k: [] for k in Kk}; O["rank_layers"] = []; cnt = {"n_frames": 0, "n_dec": 0}


def rank_of(v, j, valid):
    vv = torch.where(valid, v, torch.full_like(v, -1e9))
    return int((vv > vv[j]).sum())


with torch.no_grad():
    for bi, data in enumerate(dl):
        rgb = data["rgb"].to(dev, dtype=torch.float32); lidar = data["lidar"].to(dev, dtype=torch.float32)
        tp = data["target_point"].to(dev, dtype=torch.float32)
        vel = data["speed"].to(dev, dtype=torch.float32).unsqueeze(1); cmd = data["command"].to(dev, dtype=torch.float32)
        st = data["coop_states"].to(dev, dtype=torch.float32); mk = data["coop_mask"].to(dev, dtype=torch.float32)
        cap["on"] = True; cap["att"] = []
        o = net(rgb=rgb, lidar_bev=lidar, target_point=tp, ego_vel=vel, command=cmd, target_point_next=None, coop_states=st, coop_mask=mk)
        cap["on"] = False
        att = torch.stack(cap["att"], 1)[..., -K:]       # (bs, L, K)
        q, mem = cap["q"], cap["mem"]; S = mem.shape[1]
        ev0 = F.softmax(o[1].float(), 1) @ spd_t
        # re-run check: decoder + speed head from the captured memory reproduce the forward's E[v]
        ev_chk = F.softmax(net.target_speed_network(net.join(q, mem)[:, P]).float(), 1) @ spd_t
        if bi == 0:
            print("rerun max |dE[v]|", float((ev_chk - ev0).abs().max()), "att shape", tuple(att.shape), flush=True)
        null = net.coop_null.reshape(-1)
        dv = torch.zeros(st.shape[0], K, device=dev)
        for j in range(K):
            if not bool((mk[:, j] > 0.5).any()):
                continue
            m2 = mem.clone(); m2[:, S - K + j] = null
            ev = F.softmax(net.target_speed_network(net.join(q, m2)[:, P]).float(), 1) @ spd_t
            dv[:, j] = (ev - ev_chk).abs()
        for b in range(st.shape[0]):
            i = idx[bi * a.bs + b]; cnt["n_frames"] += 1
            m = rd(str(ds.measurements[i][0], encoding="utf-8") + f"/{ds.sample_start[i] + cur:04}.json.gz")
            oid = m.get("speed_reduced_by_obj_id")
            if oid is None:
                continue
            cnt["n_dec"] += 1
            boxes = rd(str(ds.boxes[i][cur], encoding="utf-8"))
            cands = []
            for bx in boxes:
                if bx.get("class") != "car":
                    continue
                x, y = float(bx["position"][0]), float(bx["position"][1]); d = math.hypot(x, y)
                if d <= R:
                    cands.append((d, x, y, bx.get("id"), bx.get("speed") or 0.0))
            cands.sort(key=lambda t: t[0]); cands = cands[:K]
            js = [jj for jj, c in enumerate(cands) if c[3] == oid]
            if not js:
                continue
            j = js[0]; valid = mk[b] > 0.5; nv = int(valid.sum())
            err = float(np.hypot(float(st[b, j, 0]) * 32.0 - cands[j][1], float(st[b, j, 1]) * 32.0 - cands[j][2]))
            al, am = att[b, -1], att[b].mean(0)
            O["slot"].append(j); O["nvalid"].append(nv); O["slot_err"].append(err)
            O["rank_last"].append(rank_of(al, j, valid)); O["rank_mean"].append(rank_of(am, j, valid))
            O["rank_layers"].append([rank_of(att[b, l], j, valid) for l in range(att.shape[1])])
            O["share_last"].append(float(al[j] / al[valid].sum())); O["share_mean"].append(float(am[j] / am[valid].sum()))
            O["rank_infl"].append(rank_of(dv[b], j, valid)); O["dv_dec"].append(float(dv[b, j]))
            O["dv_max"].append(float(dv[b][valid].max()))
            O["att_infl_same"].append(int(torch.where(valid, al, -1e9 * torch.ones_like(al)).argmax() == torch.where(valid, dv[b], -1e9 * torch.ones_like(al)).argmax()))
            O["dspeed"].append(float(cands[j][4])); O["dx"].append(cands[j][1]); O["dy"].append(cands[j][2])
        if bi % 50 == 0:
            print(f"batch {bi} frames {cnt['n_frames']} decisive {cnt['n_dec']} in-slots {len(O['slot'])}", flush=True)

D = {k: np.asarray(v) for k, v in O.items()}; D["n_frames"] = np.asarray([cnt["n_frames"]]); D["n_dec"] = np.asarray([cnt["n_dec"]])
if len(D["rank_layers"]) == 0:
    D["rank_layers"] = np.zeros((0, len(net.join.layers)), dtype=np.int64)
np.savez_compressed(f"{a.out}_shard{a.shard}.npz", **D)
print(json.dumps(report(D), indent=1)); print("DECISIVE_SHARD_DONE")
