"""Index of the frames with a connected hidden hazard (the 'occ' stratum of tools/v2x_offline_rate_sweep.py) over the whole
training dataset, from the box files only (no images / lidar). Same definition as the sweep: at rate 1, some slot is
valid & hidden (<= v2x_hidden_pts ego-lidar points) & hazard (moving car ahead within v2x_hazard_range m, |lat| <= 12 m),
with no augmentation. Writes <out>.npy (dataset indices, CARLA_Data(setting='all') order) and prints the counts.
  python tools/v2x_occ_index.py --root <data_root> --out /work/.../tmp/occ_index
"""
import argparse, gzip, os, sys
from multiprocessing import Pool
import numpy as np, ujson
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "team_code"))
import config as cfgmod, data as datamod, v2x_features

ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--workers", type=int, default=48)
a = ap.parse_args()
cfg = cfgmod.GlobalConfig(); cfg.initialize(root_dir=[a.root], setting="all", use_v2x=1)
ds = datamod.CARLA_Data(root=cfg.data_roots, config=cfg, estimate_class_distributions=False, estimate_sem_distribution=False, shared_dict=None, rank=0)
paths = [str(ds.boxes[i][cfg.seq_len - 1], encoding="utf-8") for i in range(len(ds))]
print(f"dataset frames {len(paths)}", flush=True)


def one(p):
    with gzip.open(p, "rt", encoding="utf-8") as f:
        boxes = ujson.load(f)
    st, mk, bk, hid, haz = v2x_features.coop_states_from_boxes(boxes, k=cfg.v2x_k, radius=cfg.v2x_radius, hidden_pts=cfg.v2x_hidden_pts,
                                                                hazard_range=cfg.v2x_hazard_range)
    return int((mk * hid * haz).sum() > 0), int(mk.sum())


with Pool(a.workers) as pool:
    r = np.array(pool.map(one, paths, chunksize=256))
occ = np.nonzero(r[:, 0])[0]
np.save(a.out + ".npy", occ)
routes = {os.path.dirname(os.path.dirname(paths[i])) for i in occ}
print(f"occluded-hazard frames {len(occ)} of {len(paths)} ({100 * len(occ) / len(paths):.1f}%) from {len(routes)} routes; mean filled slots {r[:, 1].mean():.1f}")
print("OCC_INDEX_DONE", flush=True)
