"""Checks for the inference-time message model (hidden/visible filter, noise, drop, latency) without CARLA.
  python tools/v2x_agent_filter_smoke.py --root /work/gn21/n21001/carla_garage_smoke_root
1) points_in_box reproduces the recorded num_points of the training boxes (same lidar, same box pose) -> the closed-loop
   'hidden' definition equals the training one.  2) hidden/visible partition, noise magnitude, drop rate, latency selection."""
import argparse, os, sys, glob, gzip, json, math, collections
import numpy as np, torch, laspy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "team_code"))
import v2x_features as V
ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--frames", type=int, default=40); a = ap.parse_args()
def ok(c, m):
    print(("PASS " if c else "FAIL ") + m, flush=True)
    if not c: sys.exit("FILTER SMOKE FAILED")
routes = sorted(glob.glob(os.path.join(a.root, "*", "*")))[:3]; agree = 0; total = 0; off = []
for rd in routes:
    for f in sorted(glob.glob(os.path.join(rd, "boxes", "*.json.gz")))[: a.frames // len(routes) + 1]:
        idx = os.path.basename(f)[:4]; lz = os.path.join(rd, "lidar", idx + ".laz")
        if not os.path.exists(lz): continue
        boxes = json.load(gzip.open(f, "rt")); lidar = laspy.read(lz).xyz
        for b in boxes:
            if b.get("class") != "car" or b.get("num_points") is None: continue
            n = V.points_in_box(np.array(b["position"], dtype=np.float64), float(b["yaw"]), b["extent"], lidar)
            total += 1; agree += int(n == int(b["num_points"])); off.append(abs(n - int(b["num_points"])))
off = np.array(off); print(f"recorded num_points vs points_in_box on {total} boxes from {len(routes)} routes: exact agreement {100*agree/max(total,1):.1f}%, "
      f"mean |diff| {off.mean() if len(off) else 0:.2f}, |diff|<=2 on {100*np.mean(off <= 2) if len(off) else 0:.1f}% (the .laz stores quantised points, so boundary hits can flip)")
ok(total > 50 and np.mean(off <= 2) > 0.97, "closed-loop points_in_box matches the recorded num_points within 2 hits on >97% of boxes")
# hidden flag agreement at the training threshold
hid_ok = 0; hid_n = 0
for rd in routes[:1]:
    for f in sorted(glob.glob(os.path.join(rd, "boxes", "*.json.gz")))[:20]:
        idx = os.path.basename(f)[:4]; lz = os.path.join(rd, "lidar", idx + ".laz"); boxes = json.load(gzip.open(f, "rt")); lidar = laspy.read(lz).xyz
        for b in boxes:
            if b.get("class") != "car" or b.get("num_points") is None: continue
            n = V.points_in_box(np.array(b["position"]), float(b["yaw"]), b["extent"], lidar); hid_n += 1; hid_ok += int((n <= 5) == (int(b["num_points"]) <= 5))
ok(hid_n > 0 and hid_ok / hid_n > 0.98, f"hidden flag (<=5 pts) agrees with training on {hid_ok}/{hid_n} boxes")
# synthetic scene through coop_states_from_world: 6 vehicles, 3 with lidar hits (visible) and 3 without (hidden)
def mat(x, y, yaw):
    m = np.eye(4); c, s = math.cos(yaw), math.sin(yaw); m[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]; m[:3, 3] = [x, y, 0]; return m
ego = mat(0, 0, 0); vehs = []; pts = []
for i in range(6):
    x, y = 10.0 + 6 * i, (-4.0 if i % 2 else 4.0); vehs.append((100 + i, mat(x, y, 0.3 * i), 0.3 * i, 5.0 + i, 4.5, (2.25, 1.0, 0.8)))
    if i < 3: pts.append(np.column_stack([np.random.uniform(x - 1.5, x + 1.5, 30), np.random.uniform(y - 0.6, y + 0.6, 30), np.zeros(30)]))
lidar = np.concatenate(pts, 0); rel = lambda e, m: m[:3, 3] - e[:3, 3]
st_all, mk_all = V.coop_states_from_world(ego, 0.0, vehs, k=16, relative_transform=rel, lidar=lidar, tokens="all")
S = {}; st_h, mk_h = V.coop_states_from_world(ego, 0.0, vehs, k=16, relative_transform=rel, lidar=lidar, tokens="hidden", stats=S)
st_v, mk_v = V.coop_states_from_world(ego, 0.0, vehs, k=16, relative_transform=rel, lidar=lidar, tokens="visible")
ok(int(mk_all.sum()) == 6 and int(mk_h.sum()) == 3 and int(mk_v.sum()) == 3 and S["hidden"] == 3, f"hidden/visible partition: all 6, hidden {int(mk_h.sum())}, visible {int(mk_v.sum())} (stats {S})")
xs_h = sorted(st_h[0, :3, 0].tolist()); ok(all(x * 32 > 25 for x in xs_h), "hidden set = the 3 far vehicles without lidar hits")
rng = np.random.default_rng(0); st_n, _ = V.coop_states_from_world(ego, 0.0, vehs, k=16, relative_transform=rel, tokens="all", pos_noise=1.0, rng=rng)
d = ((st_n[0, :6, :2] - st_all[0, :6, :2]) * 32).norm(dim=1); ok(0.3 < float(d.mean()) < 3.0, f"position noise sigma=1 m -> mean displacement {float(d.mean()):.2f} m")
kept = [int(V.coop_states_from_world(ego, 0.0, vehs, k=16, relative_transform=rel, drop=0.5, rng=np.random.default_rng(s))[1].sum()) for s in range(200)]
ok(2.0 < np.mean(kept) < 4.0, f"drop=0.5 keeps {np.mean(kept):.2f}/6 messages on average")
h = collections.deque(maxlen=5); ok(V.select_delayed(h, 4) is None, "latency: nothing received yet -> None")
for t in range(6): h.append(t)
ok(V.select_delayed(h, 4) == 1 and V.select_delayed(h, 0) == 5, f"latency 4 frames returns the record 4 frames old ({V.select_delayed(h, 4)}), 0 -> newest")
print("FILTER SMOKE PASSED")
