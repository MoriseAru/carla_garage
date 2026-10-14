"""BridgeDrive assets from OUR expert data: k-means anchors and per-point normalisation statistics of the checkpoints.
  python tools/bridgedrive_anchors.py --root /work/gn21/n21001/carla_garage_data_root --out <dir> [--k 60] [--workers 64]
Reproduces anchor_utils/anchor_generation.ipynb of the released BridgeDrive code on the carla_garage (PDM-Lite) data:
every training frame's measurement `route` -> the dataset's exact label processing (pad to num_route_points, NO
augmentation, smooth_path: first point 2.5 m ahead, then 1 m spacing) -> the first predict_checkpoint_len (10) points,
flattened to 20 -> sklearn KMeans(n_clusters=k, random_state=0) in raw metres -> centres sorted by ascending cluster
frequency (as the notebook) -> <out>/anchors_k{k}_10.npy (k, 10, 2) float32.
Also writes <out>/checkpoint_stats.npy (2, 10, 2) = per-point [mean, std] (x, y), the counterpart of the x_mean / x_std
constants hard-coded in the reference head (those are LEAD-data values), and <out>/anchors_report.txt.
Frames: the dataset's own index (CARLA_Data with the training config, i.e. train towns, train_sampling_rate), every
sample; the notebook used every LEAD frame. The routes are read from the measurement files only (no images / lidar)."""
import argparse, gzip, os, sys, json, time
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "team_code"))
import config as cfgmod, data as datamod

ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--k", type=int, default=60); ap.add_argument("--workers", type=int, default=64); ap.add_argument("--max_frames", type=int, default=0)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

cfg = cfgmod.GlobalConfig(); cfg.initialize(root_dir=[a.root], setting="all")
cfg.augment_percentage = 0.0; cfg.use_color_aug = 0; cfg.lidar_aug_prob = 0.0
ds = datamod.CARLA_Data(root=cfg.data_roots, config=cfg, estimate_class_distributions=False, estimate_sem_distribution=False, shared_dict=None, rank=0)
N_ROUTE, N_CK = cfg.num_route_points, cfg.predict_checkpoint_len
items = [(str(m[0], "utf-8") if isinstance(m[0], (bytes, np.bytes_)) else str(m[0]), int(s)) for m, s in zip(ds.measurements, ds.sample_start)]   # data.py: measurements[index][0] + /{seq:04}.json.gz
if a.max_frames: items = items[: a.max_frames]
print(f"frames {len(items)} (dataset index {len(ds)}); num_route_points {N_ROUTE}, checkpoints {N_CK}, smooth_route {cfg.smooth_route}", flush=True)


def route_of(item):
  """The dataset's route label without augmentation (data.py __getitem__): pad / cut to num_route_points, smooth_path."""
  mdir, seq = item
  with gzip.open(os.path.join(mdir, f"{seq:04}.json.gz"), "rt", encoding="utf-8") as f:
    route = json.load(f)["route"]
  if len(route) < N_ROUTE:
    route = np.array(route); route = np.vstack((route, np.tile(route[-1], (N_ROUTE - len(route), 1))))
  else:
    route = np.array(route[:N_ROUTE])
  route = ds.augment_route(route, y_augmentation=0.0, yaw_augmentation=0.0)
  if cfg.smooth_route:
    route = ds.smooth_path(route)
  return np.asarray(route[:N_CK, :2], dtype=np.float64)


t0 = time.time()
with Pool(a.workers) as pool:
  routes = pool.map(route_of, items, chunksize=256)
R = np.stack(routes)                                   # (N, 10, 2)
print(f"routes {R.shape} read in {time.time() - t0:.0f} s", flush=True)
mean, std = R.mean(0), R.std(0)
d = np.linalg.norm(R, axis=-1); step = np.linalg.norm(np.diff(R, axis=1), axis=-1)
rep = [f"frames {len(R)}", f"first-point distance mean {d[:, 0].mean():.3f} (min {d[:, 0].min():.3f} max {d[:, 0].max():.3f})",
       f"spacing mean {step.mean():.3f} (min {step.min():.3f} max {step.max():.3f})",
       f"x range [{R[..., 0].min():.2f}, {R[..., 0].max():.2f}]  y range [{R[..., 1].min():.2f}, {R[..., 1].max():.2f}]",
       "per-point mean x: " + " ".join(f"{v:.3f}" for v in mean[:, 0]), "per-point std  x: " + " ".join(f"{v:.3f}" for v in std[:, 0]),
       "per-point mean y: " + " ".join(f"{v:.3f}" for v in mean[:, 1]), "per-point std  y: " + " ".join(f"{v:.3f}" for v in std[:, 1])]
from sklearn.cluster import KMeans
t0 = time.time()
km = KMeans(n_clusters=a.k, random_state=0).fit(R.reshape(len(R), -1))
counts = np.bincount(km.labels_, minlength=a.k)
order = np.argsort(counts)                             # ascending frequency, as the notebook
centres = km.cluster_centers_[order].reshape(a.k, N_CK, 2).astype(np.float32)
rep += [f"kmeans k={a.k} in {time.time() - t0:.0f} s; cluster sizes min {counts.min()} median {int(np.median(counts))} max {counts.max()}",
        f"anchor first-point distance {np.linalg.norm(centres[:, 0], axis=-1).min():.3f}-{np.linalg.norm(centres[:, 0], axis=-1).max():.3f}; "
        f"anchor x range [{centres[..., 0].min():.2f}, {centres[..., 0].max():.2f}] y range [{centres[..., 1].min():.2f}, {centres[..., 1].max():.2f}]",
        f"mean distance label -> nearest anchor {np.mean(np.min(np.linalg.norm(R[:, None] - centres[None], axis=-1).mean(-1), axis=1)):.3f} m"]
np.save(os.path.join(a.out, f"anchors_k{a.k}_{N_CK}.npy"), centres)
np.save(os.path.join(a.out, "checkpoint_stats.npy"), np.stack([mean, std]).astype(np.float32))
open(os.path.join(a.out, "anchors_report.txt"), "w").write("\n".join(rep) + "\n")
print("\n".join(rep)); print("BRIDGEDRIVE_ANCHORS_DONE", flush=True)
