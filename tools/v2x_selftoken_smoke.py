"""CPU checks for self-tokens (V2X_SELF=1) without CARLA or a GPU.
  python tools/v2x_selftoken_smoke.py
Simulates a short drive (ego on a curve, five other vehicles: following, oncoming, stationary, crossing, accelerating),
turns the TRUE boxes into TF++-layout detections [x, y, ex, ey, yaw, speed=0, brake, class, score] each frame, runs the
agent's exact order of operations (tokens of step t from the tracker -> forward -> tracker.update with step-t detections),
and checks:
  1) ego_to_world / world_to_ego invert transfuser_utils.get_relative_transform (with pitch and roll in the ego matrix)
  2) with perfect detections, the self-token set at rate 0 equals the message token set at rate 1 (position < 5 cm,
     heading < 0.5 deg, speed < 0.15 m/s after the track warm-up), i.e. no frame / sign / one-step-lag bug
  3) a detection with a 180-degree heading error on a moving car is flipped back by the tracker
  4) dedup: at rate 1 every self-token is suppressed; at partial rate tokens = senders + non-senders, no duplicates
  5) noisy detections (0.15 m, 3 deg): speed error stays bounded after warm-up
  6) class filter (walkers / lights / stop signs never become tokens) and track deletion after max_miss frames
"""
import math, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "team_code"))
import v2x_features as V


def get_relative_transform(ego_matrix, vehicle_matrix):   # verbatim copy of transfuser_utils.get_relative_transform
    relative_pos = vehicle_matrix[:3, 3] - ego_matrix[:3, 3]
    rot = ego_matrix[:3, :3].T
    return rot @ relative_pos


def ok(c, m):
    print(("PASS " if c else "FAIL ") + m, flush=True)
    if not c:
        sys.exit("SELF-TOKEN SMOKE FAILED")


def mat(x, y, yaw, pitch=0.0, roll=0.0, z=0.0):
    """CARLA Transform.get_matrix() layout (yaw/pitch/roll in radians)."""
    cy, sy, cp, sp, cr, sr = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch), math.cos(roll), math.sin(roll)
    m = np.eye(4)
    m[:3, :3] = [[cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr],
                 [sy * cp, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr],
                 [sp, -cp * sr, cp * cr]]
    m[:3, 3] = [x, y, z]
    return m


# 1) round trip
rng = np.random.default_rng(0)
err = 0.0
for _ in range(200):
    e = mat(*rng.uniform(-100, 100, 2), rng.uniform(-math.pi, math.pi), rng.uniform(-0.05, 0.05), rng.uniform(-0.03, 0.03), 1.0)
    p = np.array([*rng.uniform(-100, 100, 2), e[2, 3]])
    rel = get_relative_transform(e, np.c_[np.r_[np.eye(3), np.zeros((1, 3))], np.r_[p, 1.0]])
    wx, wy = V.ego_to_world(e, rel[0], rel[1], rel[2])
    err = max(err, abs(wx - p[0]), abs(wy - p[1]))
    x2, y2 = V.world_to_ego(e, p[0], p[1], p[2])
    err = max(err, abs(x2 - rel[0]), abs(y2 - rel[1]))
ok(err < 1e-9, f"ego_to_world / world_to_ego invert get_relative_transform (max err {err:.1e} m, pitched/rolled ego)")

DT = 0.05


def world(t):
    """Ego + five vehicles at time t (s): list of (id, x, y, yaw, speed); ego first. Ego drives a 60 m-radius curve at 9 m/s."""
    R, w = 60.0, 9.0 / 60.0
    ego = (0, R * math.sin(w * t), R * (1 - math.cos(w * t)), w * t, 9.0)
    # following car, oncoming car, stationary car, crossing car, accelerating car (1.5 m/s^2); all stay inside the
    # +-32 m detection range for the 3 s run, so every track is warm when compared
    v = [(11, 8.0 + 8.0 * t, 3.0, 0.0, 8.0),
         (12, 30.0 - 8.0 * t, -3.5, math.pi, 8.0),
         (13, 25.0, 12.0, 1.2, 0.0),
         (14, 30.0, -20.0 + 6.0 * t, math.pi / 2, 6.0),
         (15, 5.0 + 2.0 * t + 0.5 * 1.5 * t * t, 8.0, 0.0, 2.0 + 1.5 * t)]
    return ego, v


def boxes_for(ego, vehs, step, noise=0.0, flip_id=None, rng=None):
    em = mat(ego[1], ego[2], ego[3])
    dets = []
    for vid, x, y, yaw, spd in vehs:
        rel = get_relative_transform(em, mat(x, y, yaw))
        if abs(rel[0]) >= 32 or abs(rel[1]) >= 32:
            continue
        ryaw = V.wrap_angle(yaw - ego[3])
        if vid == flip_id:
            ryaw = V.wrap_angle(ryaw + math.pi)
        nx = rng.normal(0, noise) if noise else 0.0
        ny = rng.normal(0, noise) if noise else 0.0
        nyaw = rng.normal(0, math.radians(3)) if noise else 0.0
        dets.append(np.array([rel[0] + nx, rel[1] + ny, 2.25, 1.0, ryaw + nyaw, 0.0, 0.0, 0, 0.9]))
    dets.append(np.array([5.0, 5.0, 0.3, 0.3, 0.0, 1.0, 0.0, 1, 0.9]))     # walker
    dets.append(np.array([20.0, -5.0, 0.5, 0.5, 0.0, 0.0, 0.0, 3, 0.9]))   # stop sign
    return dets, em


def vehicles_tuple(ego, vehs):
    return [(vid, mat(x, y, yaw), yaw, spd, 4.5, (2.25, 1.0, 0.8)) for vid, x, y, yaw, spd in vehs]


def run(noise=0.0, flip_id=None, steps=60):
    tr = V.SelfTokenTracker(dt=DT)
    rng = np.random.default_rng(1)
    worst = dict(pos=0.0, yaw=0.0, spd=[], n=0)
    for step in range(1, steps + 1):
        ego, vehs = world(step * DT)
        em, eyaw = mat(ego[1], ego[2], ego[3]), ego[3]
        toks = tr.tokens(em, eyaw, step)                                   # tokens of this step (detections up to step-1)
        extra = [t[:5] for t in toks]
        s_self, m_self = V.coop_states_from_world(em, eyaw, vehicles_tuple(ego, vehs), rate=0.0, relative_transform=get_relative_transform,
                                                  extra_tokens=extra)
        s_msg, m_msg = V.coop_states_from_world(em, eyaw, [v for v in vehicles_tuple(ego, vehs)
                                                           if abs(get_relative_transform(em, v[1])[0]) < 32 and abs(get_relative_transform(em, v[1])[1]) < 32],
                                                rate=1.0, relative_transform=get_relative_transform)
        if step > 12:                                                       # warm-up: window of 10 positions filled
            ok_n = int(m_self.sum()) == int(m_msg.sum())
            if not ok_n:
                ok(False, f"step {step}: {int(m_self.sum())} self-tokens vs {int(m_msg.sum())} message tokens")
            k = int(m_msg.sum())
            a, b = s_self[0, :k].numpy(), s_msg[0, :k].numpy()
            # pair each self-token with its nearest message token (slot order is nearest-first, and noise can swap two
            # vehicles at almost the same distance); a pairing that is not one-to-one is itself a failure
            pair = [int(np.argmin(((b[:, :2] - r[:2]) ** 2).sum(1))) for r in a]
            if len(set(pair)) != k:
                ok(False, f"step {step}: self-tokens do not map one-to-one onto vehicles: {pair}")
            worst["swaps"] = worst.get("swaps", 0) + int(pair != list(range(k)))
            b = b[pair]
            worst["pos"] = max(worst["pos"], float(np.abs(a[:, :2] - b[:, :2]).max() * 32))
            worst["yaw"] = max(worst["yaw"], float(np.degrees(np.abs(np.arctan2(a[:, 5], a[:, 4]) - np.arctan2(b[:, 5], b[:, 4])).max())))
            worst["spd"] += list(np.hypot(a[:, 2], a[:, 3]) * 10 - np.hypot(b[:, 2], b[:, 3]) * 10)
            worst["n"] += 1
        dets, _ = boxes_for(ego, vehs, step, noise, flip_id, rng)          # the forward pass of this step produces these
        tr.update(dets, em, eyaw, step)
    return worst, tr


# 2) perfect detections
w, tr = run()
ok(w["n"] > 40 and w["pos"] < 0.05 and w["yaw"] < 0.5 and np.max(np.abs(w["spd"])) < 0.45 and np.median(np.abs(w["spd"])) < 0.01,
   f"perfect detections: self-tokens at rate 0 == message tokens at rate 1 over {w['n']} frames "
   f"(max pos err {w['pos']:.3f} m, heading {w['yaw']:.2f} deg, |speed err| max {np.max(np.abs(w['spd'])):.3f} m/s; "
   f"median {np.median(np.abs(w['spd'])):.4f}; the max is the accelerating car: the regression centre lags the token step by "
   f"4.5 + 1 frames = 0.275 s x 1.5 m/s^2 = 0.41 m/s)")

# 3) flipped heading on a moving car
w, tr = run(flip_id=11)
ok(w["yaw"] < 0.5 and tr.n_flipped > 0, f"180-degree heading error on the following car is corrected ({tr.n_flipped} token-frames flipped; max heading err {w['yaw']:.2f} deg)")

# 4) dedup at rate 1 and partial rate
ego, vehs = world(3.0)
em, eyaw = mat(ego[1], ego[2], ego[3]), ego[3]
vt = vehicles_tuple(ego, vehs)
extra = []
for vid, m_, yaw, spd, ln, ext in vt:
    rel = get_relative_transform(em, m_)
    extra.append((rel[0] + 0.3, rel[1] - 0.2, V.wrap_angle(yaw - eyaw), spd, ln))
st = {}
s1, m1 = V.coop_states_from_world(em, eyaw, vt, rate=1.0, relative_transform=get_relative_transform, extra_tokens=extra, stats=st)
ok(st["self_added"] == 0 and st["self_suppressed"] == len(extra) and int(m1.sum()) == len(vt),
   f"rate 1: all {len(extra)} self-tokens suppressed, tokens = {int(m1.sum())} messages")
for rate in (0.3, 0.5, 0.7):
    st = {}
    s, m = V.coop_states_from_world(em, eyaw, vt, rate=rate, relative_transform=get_relative_transform, extra_tokens=extra, stats=st, bucket_salt=7)
    ok(int(m.sum()) == len(vt) and st["self_added"] == len(vt) - st["sending"] and st["self_suppressed"] == st["sending"],
       f"rate {rate}: {st['sending']} messages + {st['self_added']} self-tokens = {int(m.sum())} tokens, one per vehicle")

# 5) noisy detections
w, tr = run(noise=0.15)
spd = np.abs(np.array(w["spd"]))
ok(np.percentile(spd, 90) < 1.5 and w["pos"] < 1.0,
   f"noisy detections (0.15 m, 3 deg): |speed err| median {np.median(spd):.2f} p90 {np.percentile(spd, 90):.2f} m/s, max pos err {w['pos']:.2f} m "
   f"({w.get('swaps', 0)} frames where noise swapped the slot order of two vehicles)")

# 6) class filter + deletion
tr = V.SelfTokenTracker(dt=DT, max_miss=2)
em = mat(0, 0, 0)
tr.update([np.array([5.0, 5.0, 0.3, 0.3, 0.0, 1.0, 0.0, 1, 0.9]), np.array([9.0, 0.0, 2.2, 1.0, 0.0, 0.0, 0.0, 0, 0.9]),
           np.array([15.0, 2.0, 2.4, 1.0, 0.0, 0.0, 0.0, 4, 0.9])], em, 0.0, 1)
ok(len(tr.tokens(em, 0.0, 2)) == 2, "walker dropped, car (class 0) and emergency vehicle (class 4) kept")
for s in (2, 3, 4):
    tr.update([], em, 0.0, s)
ok(len(tr.tokens(em, 0.0, 5)) == 0, "tracks without a detection for more than max_miss=2 frames are gone")
tr2 = V.SelfTokenTracker(dt=DT, max_miss=2)
tr2.update([np.array([9.0, 0.0, 2.2, 1.0, 0.0, 0.0, 0.0, 0, 0.9])], em, 0.0, 1)
tr2.update([], em, 0.0, 2); tr2.update([], em, 0.0, 3)
ok(len(tr2.tokens(em, 0.0, 4)) == 1, "a track missed for 2 frames still emits a (coasted) token")
print("SELF-TOKEN SMOKE PASSED")
