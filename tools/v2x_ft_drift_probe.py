"""Read-only: decoder fine-tunes vs their source model (LOCAL HANDOVER 11043).
For each run in --runs (':'-separated run dirs; qsub -v splits on ','), on frames perm[offset : offset + n] of the seed-0
permutation (augmentation off), all models with GT tokens:
  (a) target-speed accuracy, E[v] MAE vs the expert, checkpoint L1 (same definitions as v2x_token_path_speed_probe.py);
  (c) output difference vs --ref: mean |E[v] - E[v]_ref|, share of frames whose max lateral checkpoint difference > 0.5 m.
(b) parameter drift vs --ref: ||theta - theta_ref|| / ||theta_ref|| over the run's trainable tensors (args.txt finetune.trainable),
    total and per module; computed on CPU in shard 0.
"""
import argparse, glob, json, os, sys
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "team_code"))
import jsonpickle, jsonpickle.ext.numpy as jpn; jpn.register_handlers()  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="/work/gn21/n21001/carla_garage_data_root")
ap.add_argument("--ref", default="/work/gn21/n21001/carla_garage_runs/tfpp_v2x_000")
ap.add_argument("--runs", required=True)
ap.add_argument("--offset", type=int, default=24000); ap.add_argument("--n", type=int, default=20000); ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshard", type=int, default=1)
ap.add_argument("--out", required=True); ap.add_argument("--merge", type=int, default=0)
a = ap.parse_args()
runs = [r for r in a.runs.replace(",", ":").split(":") if r]
names = ["ref"] + [os.path.basename(r.rstrip("/")) for r in runs]
GROUPS = ("join.", "checkpoint_decoder.", "target_speed_network.", "coop_proj.")


def report(D, drift):
    res = {}
    for nm in names:
        r = dict(ts_acc=float(D[nm + "_ok"].mean()), speed_mae=float(np.abs(D[nm + "_ev"] - D["v_exp"]).mean()),
                 ckpt_l1=float(D[nm + "_l1"].mean()))
        if nm != "ref":
            r["mean_abs_dEv_vs_ref"] = float(D[nm + "_dv"].mean()); r["frac_dEv_gt1_vs_ref"] = float((D[nm + "_dv"] > 1.0).mean())
            r["frac_maxlat_gt05_vs_ref"] = float((D[nm + "_dlat"] > 0.5).mean()); r["max_lat_vs_ref"] = float(D[nm + "_dlat"].max())
            r["drift"] = drift.get(nm)
        res[nm] = r
        print(nm, json.dumps(r), flush=True)
    return dict(frames=int(len(D["v_exp"])), offset=a.offset, ref=a.ref, models=res)


if a.merge:
    fs = sorted(glob.glob(a.out + "_shard*.npz")); parts = [dict(np.load(f)) for f in fs]
    D = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    drift = json.load(open(a.out + "_drift.json"))
    json.dump(report(D, drift) | {"shards": fs}, open(a.out + ".json", "w"), indent=1); print("DRIFT_MERGED"); sys.exit(0)

import config as cfgmod, model as modmod, data as datamod  # noqa: E402
dev = "cuda"


def load(run_dir):
    cfg = cfgmod.GlobalConfig(); saved = jsonpickle.decode(open(os.path.join(run_dir, "config.json")).read()); cfg.__dict__.update(saved.__dict__)
    net = modmod.LidarCenterNet(cfg); sd = torch.load(os.path.join(run_dir, "model_0030.pth"), map_location="cpu")
    net.load_state_dict(sd, strict=True); print(run_dir, "strict ok", flush=True)
    return cfg, net.to(dev).eval(), sd


cfg, net_ref, sd_ref = load(a.ref)
nets = {"ref": net_ref}; drift = {}
for r, nm in zip(runs, names[1:]):
    _, nets[nm], sd = load(r)
    tr = json.load(open(os.path.join(r, "args.txt")))["finetune"]["trainable"]
    def rel(keys):
        num = sum(float((sd[k].float() - sd_ref[k].float()).pow(2).sum()) for k in keys)
        den = sum(float(sd_ref[k].float().pow(2).sum()) for k in keys)
        return (num ** 0.5) / max(den ** 0.5, 1e-12)
    d = {"total": rel(tr), "n_trainable": len(tr)}
    for g in GROUPS:
        d[g.rstrip(".")] = rel([k for k in tr if k.startswith(g)])
    d["other"] = {k: rel([k]) for k in tr if not k.startswith(GROUPS)}
    d["frozen_changed"] = sorted(k for k in sd if k not in tr and not torch.equal(sd[k], sd_ref[k]))
    drift[nm] = d; print("drift", nm, json.dumps(d), flush=True)
    del sd
if a.shard == 0:
    json.dump(drift, open(a.out + "_drift.json", "w"), indent=1)

cfg.initialize(root_dir=[a.root], setting="all", use_v2x=1, use_v2x_bev=0)
cfg.augment_percentage = 0.0; cfg.use_color_aug = 0; cfg.lidar_aug_prob = 0.0
ds = datamod.CARLA_Data(root=cfg.data_roots, config=cfg, estimate_class_distributions=False, estimate_sem_distribution=False, shared_dict=None, rank=0)
idx = torch.randperm(len(ds), generator=torch.Generator().manual_seed(0))[a.offset: a.offset + a.n].tolist()[a.shard:: a.nshard]
dl = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, idx), batch_size=a.bs, shuffle=False, num_workers=16, drop_last=True)
print(f"dataset {len(ds)} frames; shard {a.shard}/{a.nshard}: {len(idx)} frames", flush=True)
spd_t = torch.tensor(cfg.target_speeds, dtype=torch.float32, device=dev)
K = {"v_exp": []} | {nm + s: [] for nm in names for s in ("_l1", "_ok", "_ev")} | {nm + s: [] for nm in names[1:] for s in ("_dv", "_dlat")}
out_path = f"{a.out}_shard{a.shard}.npz"
with torch.no_grad():
    for bi, data in enumerate(dl):
        g = lambda k: data[k].to(dev, dtype=torch.float32)
        tw = g("target_speed_twohot"); ts_label = tw.argmax(1); K["v_exp"] += (tw @ spd_t).tolist()
        ck = data["route"][:, : cfg.predict_checkpoint_len].to(dev, dtype=torch.float32)
        outs = {}
        for nm in names:
            o = nets[nm](rgb=g("rgb"), lidar_bev=g("lidar"), target_point=g("target_point"), ego_vel=g("speed").unsqueeze(1),
                         command=g("command"), target_point_next=None, coop_states=g("coop_states"), coop_mask=g("coop_mask"))
            pts, pck = o[1].float(), o[2].float(); ev = F.softmax(pts, 1) @ spd_t; outs[nm] = (pck, ev)
            K[nm + "_l1"] += (pck - ck).abs().mean(dim=(1, 2)).tolist(); K[nm + "_ok"] += (pts.argmax(1) == ts_label).tolist(); K[nm + "_ev"] += ev.tolist()
        for nm in names[1:]:
            K[nm + "_dv"] += (outs[nm][1] - outs["ref"][1]).abs().tolist()
            K[nm + "_dlat"] += (outs[nm][0][..., 1] - outs["ref"][0][..., 1]).abs().amax(1).tolist()
        if bi % 100 == 0:
            print(f"batch {bi} frames {len(K['v_exp'])}", flush=True)
np.savez_compressed(out_path, **{k: np.asarray(v) for k, v in K.items()})
print("DRIFT_SHARD_DONE", out_path)
