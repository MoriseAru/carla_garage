"""Read-only offline numbers for tfpp_objq_000 (LOCAL HANDOVER 11080, last request).
On frames perm[offset : offset + n] of the seed-0 permutation (augmentation off), for each run in --runs:
  planner: target-speed accuracy, E[v] MAE vs the expert, checkpoint L1 (same definitions as v2x_ft_drift_probe.py);
           runs with use_v2x get GT state tokens, the others get none.
  object queries (runs with use_obj_query_tokens): per frame, Hungarian matching of the queries to the valid labels with
           the training cost (5 * L1(x/32, y/32, v/10) - sigmoid(exist)), as in obj_query_tokens.hungarian_loss.
           pos_err = |xy_pred - xy_gt| (m) and vel_err = |v_pred - v_gt| (m/s) over matched pairs;
           existence: a query is predicted present when sigmoid(exist) > 0.5; TP = present and matched,
           FP = present and unmatched, FN = matched but not present. F1 = 2TP / (2TP + FP + FN).
"""
import argparse, glob, json, os, sys
import numpy as np, torch, torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "team_code"))
import jsonpickle, jsonpickle.ext.numpy as jpn; jpn.register_handlers()  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="/work/gn21/n21001/carla_garage_data_root")
ap.add_argument("--runs", required=True, help="':'-separated run dirs")
ap.add_argument("--offset", type=int, default=24000); ap.add_argument("--n", type=int, default=20000); ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshard", type=int, default=1)
ap.add_argument("--out", required=True); ap.add_argument("--merge", type=int, default=0)
a = ap.parse_args()
runs = [r for r in a.runs.replace(",", ":").split(":") if r]
names = [os.path.basename(r.rstrip("/")) for r in runs]

if a.merge:
    fs = sorted(glob.glob(a.out + "_shard*.npz")); parts = [dict(np.load(f)) for f in fs]
    D = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    res = {}
    for nm in names:
        r = dict(ts_acc=float(D[nm + "_ok"].mean()), speed_mae=float(np.abs(D[nm + "_ev"] - D["v_exp"]).mean()),
                 ckpt_l1=float(D[nm + "_l1"].mean()))
        if nm + "_pos" in D:
            tp, fp, fn = (int(D[nm + s].sum()) for s in ("_tp", "_fp", "_fn"))
            pos, vel = D[nm + "_pos"], D[nm + "_vel"]
            r |= dict(matched=int(len(pos)), pos_err_mean=float(pos.mean()), pos_err_median=float(np.median(pos)),
                      vel_err_mean=float(vel.mean()), vel_err_median=float(np.median(vel)),
                      tp=tp, fp=fp, fn=fn, precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1),
                      f1=2 * tp / max(2 * tp + fp + fn, 1),
                      frames_no_gt=int((D["n_gt"] == 0).sum()), gt_per_frame=float(D["n_gt"].mean()))
        res[nm] = r; print(nm, json.dumps(r), flush=True)
    json.dump(dict(frames=int(len(D["v_exp"])), offset=a.offset, n=a.n, models=res, shards=fs), open(a.out + ".json", "w"), indent=1)
    print("OBJQ_MERGED"); sys.exit(0)

import config as cfgmod, model as modmod, data as datamod  # noqa: E402
dev = "cuda"
nets, cfgs = {}, {}
for r, nm in zip(runs, names):
    cfg = cfgmod.GlobalConfig(); saved = jsonpickle.decode(open(os.path.join(r, "config.json")).read()); cfg.__dict__.update(saved.__dict__)
    net = modmod.LidarCenterNet(cfg); net.load_state_dict(torch.load(os.path.join(r, "model_0030.pth"), map_location="cpu"), strict=True)
    print(r, "strict ok; use_v2x", cfg.use_v2x, "use_obj_query_tokens", getattr(cfg, "use_obj_query_tokens", 0), flush=True)
    nets[nm], cfgs[nm] = net.to(dev).eval(), cfg
qcfg = next((c for c in cfgs.values() if getattr(c, "use_obj_query_tokens", 0)), None)

# Dataset: one loader for all models; it yields both the GT state tokens and the object-query labels.
cfg = cfgmod.GlobalConfig(); cfg.__dict__.update(next(c for c in cfgs.values() if c.use_v2x).__dict__)
cfg.initialize(root_dir=[a.root], setting="all", use_v2x=1, use_v2x_bev=0)
if qcfg is not None:
    cfg.use_obj_query_tokens = 1; cfg.obj_query_n = qcfg.obj_query_n; cfg.obj_query_radius = qcfg.obj_query_radius
cfg.augment_percentage = 0.0; cfg.use_color_aug = 0; cfg.lidar_aug_prob = 0.0
ds = datamod.CARLA_Data(root=cfg.data_roots, config=cfg, estimate_class_distributions=False, estimate_sem_distribution=False, shared_dict=None, rank=0)
idx = torch.randperm(len(ds), generator=torch.Generator().manual_seed(0))[a.offset: a.offset + a.n].tolist()[a.shard:: a.nshard]
dl = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, idx), batch_size=a.bs, shuffle=False, num_workers=16, drop_last=True)
print(f"dataset {len(ds)} frames; shard {a.shard}/{a.nshard}: {len(idx)} frames", flush=True)
spd_t = torch.tensor(cfg.target_speeds, dtype=torch.float32, device=dev)
K = {"v_exp": [], "n_gt": []} | {nm + s: [] for nm in names for s in ("_l1", "_ok", "_ev")}
qnames = [nm for nm in names if getattr(cfgs[nm], "use_obj_query_tokens", 0)]
K |= {nm + s: [] for nm in qnames for s in ("_pos", "_vel", "_tp", "_fp", "_fn")}
with torch.no_grad():
    for bi, data in enumerate(dl):
        g = lambda k: data[k].to(dev, dtype=torch.float32)
        tw = g("target_speed_twohot"); ts_label = tw.argmax(1); K["v_exp"] += (tw @ spd_t).tolist()
        ck = data["route"][:, : cfg.predict_checkpoint_len].to(dev, dtype=torch.float32)
        if qnames:
            gst, gmk = g("objq_state"), g("objq_mask"); K["n_gt"] += (gmk > 0.5).sum(1).tolist()
        for nm in names:
            v2x = bool(cfgs[nm].use_v2x)
            o = nets[nm](rgb=g("rgb"), lidar_bev=g("lidar"), target_point=g("target_point"), ego_vel=g("speed").unsqueeze(1),
                         command=g("command"), target_point_next=None,
                         coop_states=g("coop_states") if v2x else None, coop_mask=g("coop_mask") if v2x else None)
            pts, pck = o[1].float(), o[2].float(); ev = F.softmax(pts, 1) @ spd_t
            K[nm + "_l1"] += (pck - ck).abs().mean(dim=(1, 2)).tolist(); K[nm + "_ok"] += (pts.argmax(1) == ts_label).tolist(); K[nm + "_ev"] += ev.tolist()
            if nm in qnames:
                st, ex = nets[nm]._objq["state"].float(), torch.sigmoid(nets[nm]._objq["exist_logit"].float())
                cost = 5.0 * torch.cdist(st, gst, p=1) - ex[..., None]
                for b in range(st.shape[0]):
                    present = (ex[b] > 0.5).cpu().numpy(); matched = np.zeros(st.shape[1], bool)
                    valid = torch.nonzero(gmk[b] > 0.5).squeeze(1)
                    if valid.numel():
                        qi, gj = linear_sum_assignment(cost[b][:, valid].cpu().numpy()); matched[qi] = True
                        gj = valid[torch.as_tensor(gj, device=dev)]; qt = torch.as_tensor(qi, device=dev)
                        K[nm + "_pos"] += (torch.linalg.norm(st[b, qt, :2] - gst[b, gj, :2], dim=-1) * 32.0).tolist()
                        K[nm + "_vel"] += ((st[b, qt, 2] - gst[b, gj, 2]).abs() * 10.0).tolist()
                    K[nm + "_tp"].append(int((present & matched).sum())); K[nm + "_fp"].append(int((present & ~matched).sum()))
                    K[nm + "_fn"].append(int((~present & matched).sum()))
        if bi % 100 == 0:
            print(f"batch {bi} frames {len(K['v_exp'])}", flush=True)
out_path = f"{a.out}_shard{a.shard}.npz"
np.savez_compressed(out_path, **{k: np.asarray(v) for k, v in K.items()})
print("OBJQ_SHARD_DONE", out_path)
