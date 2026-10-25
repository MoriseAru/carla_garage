"""Does a lane-change command in the PDM-Lite data predict where the expert actually goes, and when?
  python tools/lanechange_command_audit.py --root /work/gn21/n21001/carla_garage_data_root [--routes 800]
For every maximal run of command 5 (change left) / 6 (change right) in a route: run length (frames at 4 Hz -> s), ego speed
at the start, and the expert's lateral displacement in the START frame's ego coordinates (x forward, y right; from
ego_matrix / pos_global) after 1, 2, 4, 6, 8 s. Answers: is the command consistent with the eventual lateral move, how
long are the windows, and how much of the move happens inside the window vs after it.
"""
import argparse, glob, gzip, json, os, random
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--routes", type=int, default=800); a = ap.parse_args()
HZ = 4.0
routes = sorted(glob.glob(os.path.join(a.root, "*", "*", "measurements"))); random.Random(0).shuffle(routes); routes = routes[: a.routes]
rows = []
for r in routes:
  ms = []
  for f in sorted(glob.glob(os.path.join(r, "*.json.gz"))):
    try: ms.append(json.load(gzip.open(f, "rt")))
    except Exception: ms.append(None)
  i = 0
  while i < len(ms):
    m = ms[i]
    if m is None or m.get("command") not in (5, 6): i += 1; continue
    c = m["command"]; j = i
    while j + 1 < len(ms) and ms[j + 1] is not None and ms[j + 1].get("command") == c: j += 1
    E = np.array(m["ego_matrix"], dtype=np.float64); Einv = np.linalg.inv(E)
    def lat(k):
      if k >= len(ms) or ms[k] is None: return np.nan
      p = np.array(list(ms[k]["pos_global"][:2]) + [0.0, 1.0]); p[2] = E[2, 3]
      return float((Einv @ p)[1])
    disp = [lat(i + int(t * HZ)) for t in (1, 2, 4, 6, 8)]
    rows.append((c, (j - i + 1) / HZ, float(m["speed"]), lat(j), *disp))
    i = j + 1
R = np.array(rows, dtype=np.float64)
print(f"routes {len(routes)}; lane-change command windows {len(R)} (left {int((R[:, 0] == 5).sum())}, right {int((R[:, 0] == 6).sum())})")
for c, name, sgn in ((5, "change LEFT (expect y < 0)", -1), (6, "change RIGHT (expect y > 0)", 1)):
  S = R[R[:, 0] == c]
  if not len(S): continue
  dur = S[:, 1]; v = S[:, 2]
  print(f"\n{name}: n={len(S)}; window length s: median {np.median(dur):.2f}, p10 {np.percentile(dur, 10):.2f}, p90 {np.percentile(dur, 90):.2f}; "
        f"share of windows <= 1 s {100 * np.mean(dur <= 1.0):.0f}%; start speed median {np.median(v):.1f} m/s")
  print(f"  lateral y at the END of the window: median {np.nanmedian(S[:, 3]):+.2f} m")
  for k, t in enumerate((1, 2, 4, 6, 8)):
    y = S[:, 4 + k]; ok = ~np.isnan(y)
    print(f"  after {t} s: median y {np.nanmedian(y):+.2f} m; moved > 2 m toward the command {100 * np.mean(sgn * y[ok] > 2):.0f}%, "
          f"> 2 m the OTHER way {100 * np.mean(-sgn * y[ok] > 2):.0f}%, |y| < 1 m {100 * np.mean(np.abs(y[ok]) < 1):.0f}% (n {int(ok.sum())})")
  short = S[S[:, 1] <= 1.0]
  if len(short):
    y4 = short[:, 6]; ok = ~np.isnan(y4)
    print(f"  SHORT windows (<= 1 s, n={len(short)}): after 4 s moved > 2 m toward the command {100 * np.mean(sgn * y4[ok] > 2):.0f}%, |y| < 1 m {100 * np.mean(np.abs(y4[ok]) < 1):.0f}%")
print("LANECHANGE_COMMAND_AUDIT_DONE")
