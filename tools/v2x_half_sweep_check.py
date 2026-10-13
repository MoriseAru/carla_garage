"""How often does the HALF-sweep lidar flag a vehicle as hidden that is visible on the 360-degree cloud? Splits the
recorded (360) .laz by azimuth into two halves. Run: python tools/v2x_half_sweep_check.py (2026-10-13: 35-50%)."""
import sys, os, glob, gzip, json, math, numpy as np, laspy
sys.path.insert(0, "/work/gn21/n21001/carla_garage/team_code"); import v2x_features as V
routes = sorted(glob.glob("/work/gn21/n21001/carla_garage_smoke_root/*/*"))[:3]
res = {}
for split in (0, 45, 90, 135):
    vis = flagged = hid = 0
    for rd in routes:
        for f in sorted(glob.glob(os.path.join(rd, "boxes", "*.json.gz")))[:15]:
            lz = os.path.join(rd, "lidar", os.path.basename(f)[:4] + ".laz")
            if not os.path.exists(lz): continue
            boxes = json.load(gzip.open(f, "rt")); pts = laspy.read(lz).xyz
            az = (np.degrees(np.arctan2(pts[:, 1], pts[:, 0])) - split) % 360
            halves = [pts[az < 180], pts[az >= 180]]
            for b in boxes:
                if b.get("class") != "car" or b.get("num_points") is None: continue
                if math.hypot(*b["position"][:2]) > 32: continue
                full = V.points_in_box(np.array(b["position"]), float(b["yaw"]), b["extent"], pts)
                if full <= 5: hid += 1; continue
                for h in halves:
                    vis += 1; flagged += int(V.points_in_box(np.array(b["position"]), float(b["yaw"]), b["extent"], h) <= 5)
    print(f"split at {split:3d} deg: visible-in-360 vehicle-halves {vis}, flagged hidden on the half sweep {100*flagged/max(vis,1):.1f}%; truly hidden {hid}")
