"""How much of a target-speed label does the ego's CURRENT speed already determine? On the TF++ (PDM-Lite) training data,
for two label definitions on the same frames:
  tfpp : TF++'s label = the expert's commanded target speed (class 0 when the expert brakes), argmax of the two-hot
  vad  : VAD round-2/3 label = the expert's mean speed over the next 1 s (from the recorded speeds, 4 Hz -> 4 frames ahead)
Trivial predictor = the class nearest to the current speed. Also split by the expert's hazard flags (vehicle_hazard /
speed_reduced_by_obj_type), the frames where an object, not the ego's inertia, sets the label.
  python tools/tfpp_label_trivial.py --root /work/gn21/n21001/carla_garage_data_root [--routes 600] [--stride 5]
"""
import argparse, glob, gzip, json, os, random
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--routes", type=int, default=600)
ap.add_argument("--stride", type=int, default=5); a = ap.parse_args()
TS = np.array([0.0, 4.0, 8.0, 10, 13.88888888, 16, 17.77777777, 20])
cls = lambda v: int(np.abs(TS - v).argmin())
def tfpp_cls(m):
  if m["brake"]: return 0
  t = float(m["target_speed"])
  if not np.any(TS > t): return len(TS) - 1
  u = int(np.argmax(TS > t)); l = u - 1
  return u if (t - TS[l]) / (TS[u] - TS[l]) > 0.5 else l
routes = sorted(glob.glob(os.path.join(a.root, "*", "*", "measurements")))
random.Random(0).shuffle(routes); routes = routes[: a.routes]
rows = []
for r in routes:
  fs = sorted(glob.glob(os.path.join(r, "*.json.gz")))
  ms = []
  for f in fs:
    try: ms.append(json.load(gzip.open(f)))
    except Exception: ms.append(None)
  for i in range(0, len(ms) - 4, a.stride):
    m = ms[i]; fut = ms[i + 1: i + 5]
    if m is None or any(x is None for x in fut): continue
    v = float(m["speed"]); v1 = float(np.mean([float(x["speed"]) for x in fut]))
    obj = bool(m.get("vehicle_hazard"))   # the expert brakes for a vehicle on this frame
    rows.append((cls(v), tfpp_cls(m), cls(v1), obj))
R = np.array(rows); now, lt, lv, obj = R[:, 0], R[:, 1], R[:, 2], R[:, 3].astype(bool)
print(f"routes {len(routes)}, frames {len(R)} (stride {a.stride}); vehicle-hazard frames {obj.sum()} ({100 * obj.mean():.1f}%)")
for name, lab in (("tfpp target speed", lt), ("vad 1-s future mean speed", lv)):
  print(f"  {name:26s}: class of the current speed is correct on {100 * (now == lab).mean():.1f}% of all frames;"
        f" vehicle hazard {100 * (now[obj] == lab[obj]).mean():.1f}%; others {100 * (now[~obj] == lab[~obj]).mean():.1f}%")
print("TFPP_LABEL_TRIVIAL_DONE")
