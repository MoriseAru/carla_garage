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
# ---- randomisation: count / distance / speed preserved, position destroyed, subset respected ----
st_r = {}
st_rv, mk_rv = V.coop_states_from_world(ego, 0.0, vehs, k=16, relative_transform=rel, lidar=lidar, randomize="visible", salt=0, stats=st_r)
st_ra, mk_ra = V.coop_states_from_world(ego, 0.0, vehs, k=16, relative_transform=rel, lidar=lidar, randomize="all", salt=0)
st_rh, mk_rh = V.coop_states_from_world(ego, 0.0, vehs, k=16, relative_transform=rel, lidar=lidar, randomize="hidden", salt=0)
ok(int(mk_rv.sum()) == 6 and int(mk_ra.sum()) == 6 and int(mk_rh.sum()) == 6, "randomisation keeps every token (count unchanged)")
ok(st_r["randomized"] == 3, f"randomize=visible touched exactly the 3 visible vehicles (randomized={st_r['randomized']})")
d_all = (st_all[0, :6, :2] * 32).norm(dim=1); d_ra = (st_ra[0, :6, :2] * 32).norm(dim=1)
ok(torch.allclose(d_all.sort().values, d_ra.sort().values, atol=1e-3), "randomize=all preserves every vehicle's distance to the ego")
ok(torch.allclose(st_all[0, :6, 6].sort().values, st_ra[0, :6, 6].sort().values, atol=1e-5), "randomize=all preserves the reported sizes")
moved = ((st_ra[0, :6, :2] - st_all[0, :6, :2]) * 32).norm(dim=1)
ok(float(moved.min()) > 1.0, f"randomize=all moves every vehicle (min displacement {float(moved.min()):.1f} m)")
# the hidden half must be untouched when only the visible half is randomised: compare the hidden-only token sets
h_ref, _ = V.coop_states_from_world(ego, 0.0, vehs, k=16, relative_transform=rel, lidar=lidar, tokens="hidden")
h_rv, _ = V.coop_states_from_world(ego, 0.0, vehs, k=16, relative_transform=rel, lidar=lidar, tokens="hidden", randomize="visible", salt=0)
ok(torch.equal(h_ref, h_rv), "randomize=visible leaves the hidden vehicles' states bitwise unchanged")
h_rh, _ = V.coop_states_from_world(ego, 0.0, vehs, k=16, relative_transform=rel, lidar=lidar, tokens="hidden", randomize="hidden", salt=0)
ok(not torch.equal(h_ref, h_rh), "randomize=hidden does change the hidden vehicles' states")
a0, a1 = V.rotation_angle(12345, 0), V.rotation_angle(12345, 0)
ok(a0 == a1 and V.rotation_angle(12345, 1) != a0, "rotation angle is constant per actor and per salt (temporally coherent ghost, different across eval seeds)")
x2, y2, yw2 = V.rotate_about_ego(10.0, 0.0, 0.0, math.pi / 2)
ok(abs(x2) < 1e-9 and abs(y2 - 10.0) < 1e-9 and abs(yw2 - math.pi / 2) < 1e-9, "rotate_about_ego rotates position and heading together")
# ---- availability router ----
import collections
h = collections.deque(maxlen=20)
ok(V.route_use_base(h, 0.7) is False, "router: no availability history yet -> keep the plug-in")
for _ in range(20): h.append(1.0)
ok(V.route_use_base(h, 0.7) is False, "router: full availability -> plug-in")
h.clear()
for _ in range(20): h.append(0.5)
ok(V.route_use_base(h, 0.7) is True and V.route_use_base(h, 0.4) is False, "router: 0.5 availability falls back to base at thresh 0.7 but not at 0.4")
h.clear()
for i in range(20): h.append(1.0 if i < 14 else 0.0)
ok(V.route_use_base(h, 0.7) is False and V.route_use_base(h, 0.75) is True, "router averages the window (14/20 frames available) instead of reacting to single frames")
ok(V.route_use_base(h, 0.0) is False, "router disabled (thresh 0) never falls back")
# ---- the availability denominator must count vehicles that do NOT send (2026-09-20 bug: gate before the count) ----
S5 = {}; V.coop_states_from_world(ego, 0.0, vehs, k=16, relative_transform=rel, rate=0.5, stats=S5)
S0 = {}; V.coop_states_from_world(ego, 0.0, vehs, k=16, relative_transform=rel, rate=0.0, stats=S0)
S1 = {}; V.coop_states_from_world(ego, 0.0, vehs, k=16, relative_transform=rel, rate=1.0, stats=S1)
ok(S5["in_radius"] == 6 and S0["in_radius"] == 6 and S1["in_radius"] == 6, f"in_radius counts every vehicle present regardless of penetration (r0.5 {S5['in_radius']}, r0 {S0['in_radius']})")
ok(S1["sending"] == 6 and S0["sending"] == 0 and 0 < S5["sending"] < 6, f"sending follows the gate: r1 {S1['sending']}, r0.5 {S5['sending']}, r0 {S0['sending']}")
ok(S0["sending"] / S0["in_radius"] == 0.0, "r0 yields availability 0.0 (a recorded zero, so the router can fire on it)")
h0 = collections.deque([0.0] * 20, maxlen=20); ok(V.route_use_base(h0, 0.7) is True, "router falls back at r0 (history of zeros)")
Sd = {}; V.coop_states_from_world(ego, 0.0, vehs, k=16, relative_transform=rel, drop=0.5, rng=np.random.default_rng(3), stats=Sd)
ok(Sd["sending"] + Sd["dropped"] == 6 and Sd["kept"] == Sd["sending"], f"drop lowers sending (kept {Sd['kept']} = sending {Sd['sending']}, dropped {Sd['dropped']})")
Sk = {}; many = [(200 + i, mat(8.0 + 2.5 * i, 3.0, 0.0), 0.0, 5.0, 4.5, (2.25, 1.0, 0.8)) for i in range(20)]
V.coop_states_from_world(ego, 0.0, many, k=16, relative_transform=rel, stats=Sk)
ok(Sk["in_radius"] == 20 and Sk["sending"] == 20 and Sk["kept"] == 16, "the K-slot cap lowers kept but not sending: 20 senders -> availability 1.0, 16 tokens")
print("FILTER SMOKE PASSED")
