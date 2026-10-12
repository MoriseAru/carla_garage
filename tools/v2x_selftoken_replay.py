"""Offline replay of the self-token tracker on frames recorded in closed loop (V2X_SELF_DUMP=<dir> on the agent).
  python tools/v2x_selftoken_replay.py <dump_dir> [--variant base] [--breakdown]
Each dump holds, per agent step: the ego pose, the ground-truth vehicles (id, matrix, yaw, speed, length, extent) and the
NMS'd detections of that step. The replay feeds the detections to a tracker exactly as the agent does (tokens of step t are
read BEFORE the detections of step t are added) and scores the tokens like sensor_agent.v2x_self_tokens: nearest
ground-truth vehicle within 2.5 m in the ego frame -> position / speed / heading error. Detections do not depend on the
tokens (bitwise, 2026-10-13 offline check A), so a replay with a different tracker is exact, not an approximation --
except that the ego trajectory itself would differ in closed loop once the tokens change the driving.
"""
import argparse, glob, gzip, math, os, pickle, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "team_code"))
import v2x_features as V


def rel_xy(ego_matrix, actor_matrix):
    m = np.asarray(ego_matrix, dtype=np.float64); a = np.asarray(actor_matrix, dtype=np.float64)
    r = m[:3, :3].T @ (a[:3, 3] - m[:3, 3])
    return float(r[0]), float(r[1])


def replay(frames, make_tracker, match=2.5):
    tr = make_tracker()
    rows = []   # (spd_err, pos_err, yaw_err, hits, gt_speed, dist, est_speed)
    for fr in frames:
        em, ey, step = fr["ego_matrix"], fr["ego_yaw"], fr["step"]
        toks = tr.tokens(em, ey, step)
        gt = [(*rel_xy(em, g[1]), V.wrap_angle(g[2] - ey), g[3]) for g in fr["gt"]]
        for t in toks:
            x, y, ryaw, spd = t[0], t[1], t[2], t[3]
            j, dmin = -1, 1e9
            for i, g in enumerate(gt):
                d = math.hypot(g[0] - x, g[1] - y)
                if d < dmin:
                    j, dmin = i, d
            if j < 0 or dmin > match:
                continue
            g = gt[j]
            hits = tr.tracks[t[5]]["hits"] if t[5] in tr.tracks else -1
            rows.append((spd - g[3], dmin, abs(V.wrap_angle(ryaw - g[2])), hits, g[3], math.hypot(x, y), spd))
        if fr["dets"] is not None:
            tr.update(fr["dets"], em, ey, step)
    return np.array(rows, dtype=np.float64).reshape(-1, 7)


def summary(r, label):
    e = np.abs(r[:, 0])
    return (f"{label:28s} n={len(r):6d}  spd med {np.median(r[:, 0]):+.2f} |p90| {np.percentile(e, 90):.2f} "
            f"|p99| {np.percentile(e, 99):.2f}  pos med {np.median(r[:, 1]):.2f} p90 {np.percentile(r[:, 1], 90):.2f}  "
            f"flip {int((r[:, 2] > math.pi / 2).sum())}")


def load(dump_dir):
    out = []
    for f in sorted(glob.glob(os.path.join(dump_dir, "selfdump_*.pkl.gz"))):
        with gzip.open(f, "rb") as fh:
            out.append((os.path.basename(f), pickle.load(fh)))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("dump_dir"); ap.add_argument("--breakdown", action="store_true")
    a = ap.parse_args()
    routes = load(a.dump_dir)
    R = np.concatenate([replay(fr, V.SelfTokenTracker) for _, fr in routes])
    print(summary(R, "SelfTokenTracker (default)"))
    if a.breakdown:
        for lo, hi in ((1, 1), (2, 2), (3, 3), (4, 5), (6, 10), (11, 30), (31, 10 ** 9)):
            m = (R[:, 3] >= lo) & (R[:, 3] <= hi)
            if m.any():
                print(summary(R[m], f"  hits {lo}-{hi}") + f"  share of |err|>1.5: {(np.abs(R[m, 0]) > 1.5).sum() / max((np.abs(R[:, 0]) > 1.5).sum(), 1):.2f}")
        for lo, hi in ((0, 0.5), (0.5, 3), (3, 8), (8, 99)):
            m = (R[:, 4] >= lo) & (R[:, 4] < hi)
            if m.any():
                print(summary(R[m], f"  gt speed {lo}-{hi} m/s") + f"  share of |err|>1.5: {(np.abs(R[m, 0]) > 1.5).sum() / max((np.abs(R[:, 0]) > 1.5).sum(), 1):.2f}")
