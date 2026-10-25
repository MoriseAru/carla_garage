"""LOCAL 10/25 04:55 request (1): on lane-change frames (command 5 change-left / 6 change-right), how lateral are the NEAR
checkpoints of the expert label and of each model -- did the V2XState plug-in learn a flat near field, or is the label flat
and the base bends by chance?
  python tools/lanechange_nearfield.py --root /work/gn21/n21001/carla_garage_data_root --runs <dir> [<dir> ...] [--scan 60000]
Frames: a seed-0 random scan of --scan dataset indices, keeping those whose current measurement has command 5 / 6;
un-augmented (closed-loop distribution). These are training-distribution frames (TF++ trains on the whole data root).
Per model and per checkpoint index k (0-based; k = 1 is ~3.5 m ahead, the one TF++'s controller steers on at 8-10 m/s):
mean signed y toward the lane-change side (+ = toward the commanded side), mean |y|, and |y_pred - y_label|; strata: all
lane-change frames, frames whose LABEL near point already bends (|y_label[1]| > 0.1 m), and frames where only the far point
bends (|y_label[9]| > 1 m but |y_label[1]| < 0.1 m).
"""
import argparse, copy, gzip, json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "team_code"))
import jsonpickle, jsonpickle.ext.numpy as jsonpickle_numpy; jsonpickle_numpy.register_handlers()
import config as cfgmod, model as modmod, data as datamod

ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--runs", nargs="+", required=True)
ap.add_argument("--scan", type=int, default=60000); ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--out", default="/work/gn21/n21001/V2XState_Real/tmp/lanechange_nearfield.json"); a = ap.parse_args(); dev = "cuda"


def load(run_dir):
  cfg = cfgmod.GlobalConfig(); saved = jsonpickle.decode(open(os.path.join(run_dir, "config.json")).read()); cfg.__dict__.update(saved.__dict__)
  net = modmod.LidarCenterNet(cfg); ck = sorted(f for f in os.listdir(run_dir) if f.startswith("model_") and f.endswith(".pth"))[-1]
  net.load_state_dict(torch.load(os.path.join(run_dir, ck), map_location="cpu"), strict=True); net.to(dev).eval()
  print(f"loaded {run_dir}/{ck}: v2xstate_head {getattr(cfg, 'use_v2xstate_head', 0)}, use_v2x {getattr(cfg, 'use_v2x', 0)}", flush=True)
  return cfg, net


models = {os.path.basename(r.rstrip("/")): load(r) for r in a.runs}
cfg0 = copy.deepcopy(next(iter(models.values()))[0]); cfg0.initialize(root_dir=[a.root], setting="all")
cfg0.use_v2x = 1; cfg0.augment_percentage = 0.0; cfg0.use_color_aug = 0; cfg0.lidar_aug_prob = 0.0
ds = datamod.CARLA_Data(root=cfg0.data_roots, config=cfg0, estimate_class_distributions=False, estimate_sem_distribution=False, shared_dict=None, rank=0)
cand = torch.randperm(len(ds), generator=torch.Generator().manual_seed(0))[: a.scan].tolist(); keep, side = [], []
for i in cand:
  f = str(ds.measurements[i][0], encoding="utf-8") + f"/{int(ds.sample_start[i]) + cfg0.seq_len - 1:04}.json.gz"
  try: c = json.load(gzip.open(f, "rt"))["command"]
  except Exception: continue
  if c in (5, 6): keep.append(i); side.append(-1.0 if c == 5 else 1.0)   # y right: change-left = toward negative y
print(f"scanned {len(cand)} of {len(ds)} frames; lane-change frames {len(keep)} (left {side.count(-1.0)}, right {side.count(1.0)})", flush=True)
dl = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, keep), batch_size=a.bs, shuffle=False, num_workers=16)
P = cfg0.predict_checkpoint_len; lab, spd = [], []; pred = {k: [] for k in models}
with torch.no_grad():
  for data in dl:
    kw = dict(rgb=data["rgb"].to(dev, dtype=torch.float32), lidar_bev=data["lidar"].to(dev, dtype=torch.float32),
              target_point=data["target_point"].to(dev, dtype=torch.float32), ego_vel=data["speed"].to(dev, dtype=torch.float32)[:, None],
              command=data["command"].to(dev, dtype=torch.float32))
    for name, (cfg, net) in models.items():
      kk = dict(kw)
      if getattr(cfg, "use_v2x", 0):
        kk.update(coop_states=data["coop_states"].to(dev, dtype=torch.float32), coop_mask=data["coop_mask"].to(dev, dtype=torch.float32))
      pred[name].append(net(**kk)[2].float().cpu())
    lab.append(data["route"][:, :P].float()); spd.append(data["speed"].float())
L = torch.cat(lab).numpy(); S = np.array(side)[:, None]; V = torch.cat(spd).numpy()
Y = {k: torch.cat(v).numpy() for k, v in pred.items()}
strata = dict(all=np.ones(len(L), bool), label_near_bends=np.abs(L[:, 1, 1]) > 0.1, label_far_only=(np.abs(L[:, 9, 1]) > 1.0) & (np.abs(L[:, 1, 1]) < 0.1))
res = dict(n=len(L), strata={k: int(v.sum()) for k, v in strata.items()}, speed_median=float(np.median(V)), rows={})
ks = [0, 1, 2, 3, 5, 9]
print(f"\nlane-change frames {len(L)}; ego speed median {np.median(V):.1f} m/s; strata " + ", ".join(f"{k} {int(v.sum())}" for k, v in strata.items()))
print("signed lateral offset toward the commanded side y*sign (m), mean; then |y_pred - y_label| mean; checkpoints " + " ".join(f"k{k}" for k in ks))
for sname, sel in strata.items():
  if not sel.any(): continue
  print(f"\n  [{sname}] n={int(sel.sum())}")
  rows = {"label": (L[sel, :, 1] * S[sel]).mean(0)}
  for name, y in Y.items(): rows[name] = (y[sel, :, 1] * S[sel]).mean(0)
  for name, v in rows.items():
    err = "" if name == "label" else "  | err " + " ".join(f"{np.abs(Y[name][sel, k, 1] - L[sel, k, 1]).mean():.3f}" for k in ks)
    print(f"    {name:22s} " + " ".join(f"{v[k]:+.3f}" for k in ks) + err)
    res["rows"][f"{sname}|{name}"] = dict(signed=[float(v[k]) for k in ks], err=None if name == "label" else [float(np.abs(Y[name][sel, k, 1] - L[sel, k, 1]).mean()) for k in ks])
  near_far = {name: float((Y[name][sel, 1, 1] * S[sel, 0]).mean() / max((Y[name][sel, 9, 1] * S[sel, 0]).mean(), 1e-6)) for name in Y}
  near_far["label"] = float((L[sel, 1, 1] * S[sel, 0]).mean() / max((L[sel, 9, 1] * S[sel, 0]).mean(), 1e-6))
  print("    near/far ratio (k1 / k9, signed): " + ", ".join(f"{k} {v:.3f}" for k, v in near_far.items())); res["rows"][f"{sname}|near_far"] = near_far
# raw y (y right) and |y| per command, no sign convention assumed
SD = np.array(side)
print("\nRAW lateral y (m, + = right) and |y|, per command; checkpoints " + " ".join(f"k{k}" for k in ks))
for cname, cv in (("cmd5 change-left", -1.0), ("cmd6 change-right", 1.0)):
  for sname, sel0 in strata.items():
    sel = sel0 & (SD == cv)
    if not sel.any(): continue
    print(f"  [{cname} | {sname}] n={int(sel.sum())}")
    for name, y in [("label", L)] + list(Y.items()):
      print(f"    {name:22s} y " + " ".join(f"{y[sel, k, 1].mean():+.3f}" for k in ks) + "   |y| " + " ".join(f"{np.abs(y[sel, k, 1]).mean():.3f}" for k in ks))
      res["rows"][f"raw|{cname}|{sname}|{name}"] = dict(y=[float(y[sel, k, 1].mean()) for k in ks], absy=[float(np.abs(y[sel, k, 1]).mean()) for k in ks])
# frames where the label far point moves toward each side, regardless of the command
for nm, sel in (("label k9 y < -1 (path to the left)", L[:, 9, 1] < -1), ("label k9 y > +1 (path to the right)", L[:, 9, 1] > 1)):
  print(f"  [{nm}] n={int(sel.sum())}, of which cmd5 {int((sel & (SD < 0)).sum())} / cmd6 {int((sel & (SD > 0)).sum())}")
  for name, y in [("label", L)] + list(Y.items()):
    print(f"    {name:22s} |y| " + " ".join(f"{np.abs(y[sel, k, 1]).mean():.3f}" for k in ks) + "   err " + ("-" if name == "label" else " ".join(f"{np.abs(y[sel, k, 1] - L[sel, k, 1]).mean():.3f}" for k in ks)))
json.dump(res, open(a.out, "w"), indent=1); print("LANECHANGE_NEARFIELD_DONE")
