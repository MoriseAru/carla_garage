"""Why is the V2XState plug-in's near field straight at a lane-change ONSET (LOCAL 10/25 17:17)? Offline, on training frames.
  python tools/lanechange_onset.py --root /work/gn21/n21001/carla_garage_data_root --vs <plug-in run> --vb <base-head run>
     [--others <run> ...] [--routes 2500]
Frames: the first 0.75 s (3 frames at 4 Hz) of every command-5 / -6 window in a seed-0 sample of routes, where the ego has
not started moving sideways yet: |label y at checkpoint 0| < 0.3 m and the target point is 5-20 m ahead and 2-5 m to the
commanded side. (The closed-loop failure state: straight ego, lane-change command, near target point in the next lane.)
Reports, per model: y at checkpoints 0..3 and 9 vs the label, and the TARGET-POINT GAIN of each checkpoint
(d y_k / d target_y from +-1 m lateral shifts of the target point, everything else fixed).
HEAD SWAP (the two V2XState heads take inputs of identical shape): plug-in head on the base model's inputs (its BEV features,
occupancy, state) and base head on the plug-in's inputs -> is the straight near field in the head's weights or in the trunk?
"""
import argparse, collections, copy, glob, gzip, json, os, random, sys
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "team_code"))
import jsonpickle, jsonpickle.ext.numpy as jsonpickle_numpy; jsonpickle_numpy.register_handlers()
import config as cfgmod, model as modmod, data as datamod

ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--vs", required=True); ap.add_argument("--vb", required=True)
ap.add_argument("--others", nargs="*", default=[]); ap.add_argument("--routes", type=int, default=2500); ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--out", default="/work/gn21/n21001/V2XState_Real/tmp/lanechange_onset.json"); a = ap.parse_args(); dev = "cuda"


def load(run_dir):
  cfg = cfgmod.GlobalConfig(); saved = jsonpickle.decode(open(os.path.join(run_dir, "config.json")).read()); cfg.__dict__.update(saved.__dict__)
  net = modmod.LidarCenterNet(cfg); ck = sorted(f for f in os.listdir(run_dir) if f.startswith("model_") and f.endswith(".pth"))[-1]
  net.load_state_dict(torch.load(os.path.join(run_dir, ck), map_location="cpu"), strict=True); net.to(dev).eval()
  return cfg, net


names = ["plugin", "base"] + [os.path.basename(r.rstrip("/")) for r in a.others]
models = dict(zip(names, [load(r) for r in [a.vs, a.vb] + a.others]))
assert getattr(models["plugin"][0], "use_v2xstate_head", 0) and getattr(models["base"][0], "use_v2xstate_head", 0)
cfg0 = copy.deepcopy(models["base"][0]); cfg0.initialize(root_dir=[a.root], setting="all")
cfg0.use_v2x = 1; cfg0.augment_percentage = 0.0; cfg0.use_color_aug = 0; cfg0.lidar_aug_prob = 0.0
ds = datamod.CARLA_Data(root=cfg0.data_roots, config=cfg0, estimate_class_distributions=False, estimate_sem_distribution=False, shared_dict=None, rank=0)
where = {}
for i in range(len(ds)):
  where[(str(ds.measurements[i][0], encoding="utf-8"), int(ds.sample_start[i]) + cfg0.seq_len - 1)] = i
mdirs = sorted({k[0] for k in where}); random.Random(0).shuffle(mdirs); mdirs = mdirs[: a.routes]
cand, side = [], []
for md in mdirs:
  frames = sorted(int(os.path.basename(f)[:4]) for f in glob.glob(os.path.join(md, "*.json.gz")))
  prev = None
  for fr in frames:
    try: c = json.load(gzip.open(os.path.join(md, f"{fr:04}.json.gz"), "rt"))["command"]
    except Exception: c = None
    if c in (5, 6) and prev != c:          # window start: take its first 3 frames
      for k in range(3):
        if (md, fr + k) in where: cand.append(where[(md, fr + k)]); side.append(-1.0 if c == 5 else 1.0)
    prev = c
print(f"routes scanned {len(mdirs)}; window-start frames {len(cand)}", flush=True)
dl = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, cand), batch_size=a.bs, shuffle=False, num_workers=16)
cap = {}
def hook_for(name):
  def h(mod, inp, out): cap[name] = [x.detach() for x in inp]
  return h
for n in ("plugin", "base"): models[n][1].v2xstate_head.register_forward_hook(hook_for(n))
P = 10; rec = collections.defaultdict(list)
def run(name, kw, data):
  cfg, net = models[name]; kk = dict(kw)
  if getattr(cfg, "use_v2x", 0): kk.update(coop_states=data["coop_states"].to(dev, dtype=torch.float32), coop_mask=data["coop_mask"].to(dev, dtype=torch.float32))
  return net(**kk)[2].float()
with torch.no_grad():
  for data in dl:
    tp = data["target_point"].to(dev, dtype=torch.float32)
    kw = dict(rgb=data["rgb"].to(dev, dtype=torch.float32), lidar_bev=data["lidar"].to(dev, dtype=torch.float32), target_point=tp,
              ego_vel=data["speed"].to(dev, dtype=torch.float32)[:, None], command=data["command"].to(dev, dtype=torch.float32))
    for n in models:
      rec[n].append(run(n, kw, data).cpu())
      gains = []
      for d in (+1.0, -1.0):
        k2 = dict(kw); t2 = tp.clone(); t2[:, 1] += d; k2["target_point"] = t2; gains.append(run(n, k2, data)[..., 1].cpu())
      rec[n + "_gain"].append((gains[0] - gains[1]) / 2.0)
    # head swap: each V2XState head on the other model's head inputs (captured on the plain r1 forward)
    run("plugin", kw, data); ip = cap["plugin"]; run("base", kw, data); ib = cap["base"]
    hp, hb = models["plugin"][1].v2xstate_head, models["base"][1].v2xstate_head
    rec["plugin_head_on_base_inputs"].append(hp(*ib)["checkpoints"].float().cpu())
    rec["base_head_on_plugin_inputs"].append(hb(*ip)["checkpoints"].float().cpu())
    # partial swaps on the plug-in head: base BEV features only / base occupancy only
    rec["plugin_head_base_feat"].append(hp(ib[0], ip[1], ip[2], ip[3])["checkpoints"].float().cpu())
    rec["plugin_head_base_occ"].append(hp(ip[0], ib[1], ip[2], ip[3])["checkpoints"].float().cpu())
    rec["label"].append(data["route"][:, :P].float()); rec["tp"].append(tp.cpu()); rec["speed"].append(data["speed"].float())
R = {k: torch.cat(v).numpy() for k, v in rec.items()}; S = np.array(side)
L, TP = R["label"], R["tp"]
onset = (np.abs(L[:, 0, 1]) < 0.3) & (TP[:, 0] > 5) & (TP[:, 0] < 20) & (S * TP[:, 1] > 2) & (S * TP[:, 1] < 5)
res = dict(n_candidates=len(L), n_onset=int(onset.sum()), speed_median=float(np.median(R["speed"][onset])) if onset.any() else None, rows={})
ks = [0, 1, 2, 3, 9]
print(f"\nwindow-start frames {len(L)}; ONSET frames (straight ego, near target 5-20 m ahead, 2-5 m to the commanded side) {int(onset.sum())}; "
      f"speed median {res['speed_median']}; target point median ({np.median(TP[onset, 0]):.1f}, {np.median(np.abs(TP[onset, 1])):.1f}) m")
print("signed y toward the commanded side (m) at checkpoints " + " ".join(f"k{k}" for k in ks) + "   | target-point gain d y_k / d tp_y at the same k")
order = ["label"] + list(models) + ["plugin_head_on_base_inputs", "base_head_on_plugin_inputs", "plugin_head_base_feat", "plugin_head_base_occ"]
for n in order:
  y = R[n][..., 1] if R[n].ndim == 3 else R[n]
  v = (y[onset] * S[onset, None]).mean(0)
  g = "" if n + "_gain" not in R else "   | gain " + " ".join(f"{R[n + '_gain'][onset][:, k].mean():.3f}" for k in ks)
  print(f"  {n:28s} " + " ".join(f"{v[k]:+.3f}" for k in ks) + g)
  res["rows"][n] = dict(signed=[float(v[k]) for k in ks], gain=None if not g else [float(R[n + "_gain"][onset][:, k].mean()) for k in ks])
for n in ("plugin", "base"):
  frac = float(np.mean(np.abs(R[n][onset, 1, 1]) < 0.05)); res["rows"][n]["frac_k1_straight"] = frac
  print(f"  share of onset frames with |y_k1| < 0.05 m: {n} {100 * frac:.0f}%")
print(f"  share with |label y_k1| < 0.05 m: {100 * np.mean(np.abs(L[onset, 1, 1]) < 0.05):.0f}%")
json.dump(res, open(a.out, "w"), indent=1); print("LANECHANGE_ONSET_DONE")
