"""Fix B (user 2026-10-26): floor on the lateral controller's checkpoint index (config lateral_min_lookahead / env
LAT_MIN_LOOKAHEAD). Smoke + a first-order estimate on the lane-change onset that LOCAL logged (10/25 17:17, route 23670 step 84).
  python tools/lateral_floor_smoke.py
 1) floor 0 = the released controller bit for bit (same steer on random paths and speeds);
 2) the floor only acts below the speed where the speed rule already reaches it (2: ~10.8 m/s, 3: ~14 m/s);
 3) on the logged onset paths (checkpoints 2.5 / 3.5 / 4.4 m), the plug-in's first-step steer with floors 0 / 2 / 3 vs the
    V2XState base's with the released rule (the base did change lanes there).
"""
import copy, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "team_code"))
import config as cfgmod
from nav_planner import LateralPIDController


def ok(c, m):
  print(("PASS " if c else "FAIL ") + m, flush=True)
  if not c:
    sys.exit("LATERAL FLOOR SMOKE FAILED")


cfg = cfgmod.GlobalConfig()
def steer(path, v, floor):
  c = copy.deepcopy(cfg); c.lateral_min_lookahead = floor; ctl = LateralPIDController(c)
  return ctl.step(np.asarray(path, dtype=np.float64), v, np.zeros(2), 0.0)

rng = np.random.default_rng(0); same = True
for _ in range(200):
  x = 2.5 + np.arange(10.0); y = np.cumsum(rng.normal(0, 0.3, 10)); v = rng.uniform(0, 20)
  c0 = copy.deepcopy(cfg); ref = LateralPIDController(c0).step(np.stack([x, y], 1), v, np.zeros(2), 0.0)
  same &= steer(np.stack([x, y], 1), v, 0) == ref
ok(same, "floor 0 reproduces the released controller exactly (200 random paths / speeds)")
def idx(v):
  return int(np.clip(cfg.lateral_speed_scale * v * 3.6 + cfg.lateral_speed_offset, 24, 105) / 10 - 2)
v2 = next(v for v in np.arange(0, 30, 0.05) if idx(v) >= 2); v3 = next(v for v in np.arange(0, 30, 0.05) if idx(v) >= 3)
print(f"  speed rule: index 1 from {next(v for v in np.arange(0, 30, 0.05) if idx(v) >= 1):.1f} m/s, 2 from {v2:.1f}, 3 from {v3:.1f}")
path = np.stack([2.5 + np.arange(10.0), np.linspace(0, -3, 10) ** 1], 1)
ok(all(steer(path, v, 2) == steer(path, v, 0) for v in np.arange(v2 + 0.1, 25, 0.5)) and all(steer(path, v, 3) == steer(path, v, 0) for v in np.arange(v3 + 0.1, 25, 0.5)),
   f"floor 2 / 3 change nothing above {v2:.1f} / {v3:.1f} m/s")
# 3) LOCAL's logged onset (23670 step 84; y at 2.5 / 3.5 / 4.4 m; ~10 m point from the 'approx 10 m' column)
plug = [(2.5, 0.01), (3.5, -0.00), (4.4, -0.24), (10.0, -3.23)]
base = [(2.5, 0.03), (3.5, -0.18), (4.4, -0.57), (10.0, -3.38)]
def as_path(pts):
  xs = np.array([p[0] for p in pts]); ys = np.array([p[1] for p in pts]); xq = 2.5 + np.arange(10.0)
  return np.stack([xq, np.interp(xq, xs, ys)], 1)
for v in (8.0, 9.0, 10.0):
  sp = [steer(as_path(plug), v, f) for f in (0, 2, 3)]; sb = steer(as_path(base), v, 0)
  print(f"  {v:.0f} m/s: plug-in steer floor 0 / 2 / 3 = {sp[0]:+.3f} / {sp[1]:+.3f} / {sp[2]:+.3f};  V2XState base (released rule) {sb:+.3f}")
print("  (first-step steer only, linear interpolation between the logged points -- an indication, not a closed-loop result)")
print("LATERAL_FLOOR_SMOKE_PASSED")
