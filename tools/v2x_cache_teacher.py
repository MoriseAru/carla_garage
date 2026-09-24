"""Cache the co-trained plug-in's outputs at penetration 1 as a distillation TEACHER, frame-aligned with the base-feature
cache (same dataset order, same un-augmented frames, same shards).
  python tools/v2x_cache_teacher.py --root <data_root> --teacher <plugin_run_dir> --out <cache_dir> --shard 0 --nshards 4
Writes shard<k>_teacher_ts.npy (N,C) logits, shard<k>_teacher_ckpt.npy (N,L,2), shard<k>_teacher_index.npy (must equal
shard<k>_index.npy of the base cache) and shard<k>_teacher_meta.json.
"""
import argparse, os, sys, json, time, hashlib
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "team_code"))
import jsonpickle, jsonpickle.ext.numpy as jsonpickle_numpy; jsonpickle_numpy.register_handlers()
import config as cfgmod, model as modmod, data as datamod

ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--teacher", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1); ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--workers", type=int, default=48); ap.add_argument("--max_frames", type=int, default=0); a = ap.parse_args(); dev = "cuda"

def load_cfg(run_dir):
    cfg = cfgmod.GlobalConfig(); saved = jsonpickle.decode(open(os.path.join(run_dir, "config.json")).read()); cfg.__dict__.update(saved.__dict__); return cfg
cfg_d = load_cfg(a.teacher); cfg_d.initialize(root_dir=[a.root], setting="all", use_v2x=1)
cfg_d.augment_percentage = 0.0; cfg_d.use_color_aug = 0; cfg_d.lidar_aug_prob = 0.0        # identical frames to the base cache
ds = datamod.CARLA_Data(root=cfg_d.data_roots, config=cfg_d, estimate_class_distributions=False, estimate_sem_distribution=False, shared_dict=None, rank=0)
idx = np.array_split(np.arange(len(ds)), a.nshards)[a.shard]
if a.max_frames: idx = idx[: a.max_frames]
n = len(idx); L = cfg_d.predict_checkpoint_len
print(f"dataset {len(ds)} frames; shard {a.shard}/{a.nshards}: {n} frames [{idx[0]}..{idx[-1]}]", flush=True)
base_index = os.path.join(a.out, f"shard{a.shard:02d}_index.npy")
if os.path.exists(base_index):
    bi = np.load(base_index); assert bi.shape[0] == n and (bi == idx).all(), "teacher shard is not aligned with the base cache shard"
    print("aligned with the base cache shard index", flush=True)

cfg_m = load_cfg(a.teacher); assert cfg_m.use_v2x and not getattr(cfg_m, "use_v2x_adapter", 0), "teacher must be the co-trained plug-in"
net = modmod.LidarCenterNet(cfg_m); ck = sorted(f for f in os.listdir(a.teacher) if f.startswith("model_") and f.endswith(".pth"))[-1]
ck_path = os.path.join(a.teacher, ck); net.load_state_dict(torch.load(ck_path, map_location="cpu"), strict=True); net.to(dev).eval()
cfg_m.use_semantic = cfg_m.use_bev_semantic = cfg_m.use_depth = cfg_m.detect_boxes = 0
os.makedirs(a.out, exist_ok=True); pre = os.path.join(a.out, f"shard{a.shard:02d}_teacher_")
def mm(name, shape, dtype): return np.lib.format.open_memmap(pre + name + ".npy", mode="w+", dtype=dtype, shape=shape)
t_ts = mm("ts", (n, len(cfg_d.target_speeds)), np.float32); t_ck = mm("ckpt", (n, L, 2), np.float32); t_ix = mm("index", (n,), np.int64)
dl = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, idx.tolist()), batch_size=a.bs, shuffle=False, num_workers=a.workers, pin_memory=True)
t0 = time.time(); o = 0
with torch.no_grad():
    for bi_, data in enumerate(dl):
        rgb = data["rgb"].to(dev, dtype=torch.float32, non_blocking=True); lidar = data["lidar"].to(dev, dtype=torch.float32, non_blocking=True)
        tp = data["target_point"].to(dev, dtype=torch.float32); vel = data["speed"].to(dev, dtype=torch.float32).unsqueeze(1); cmd = data["command"].to(dev, dtype=torch.float32)
        st = data["coop_states"].to(dev, dtype=torch.float32); mk = data["coop_mask"].to(dev, dtype=torch.float32)       # penetration 1: every recorded vehicle sends
        out = net(rgb=rgb, lidar_bev=lidar, target_point=tp, ego_vel=vel, command=cmd, coop_states=st, coop_mask=mk)
        bs = rgb.shape[0]; sl = slice(o, o + bs)
        t_ts[sl] = out[1].float().cpu().numpy(); t_ck[sl] = out[2].float().cpu().numpy(); t_ix[sl] = idx[o:o + bs]; o += bs
        if bi_ % 100 == 0 or bi_ == 10: print(f"  {o}/{n} frames, {o / (time.time() - t0):.1f} frames/s", flush=True)
assert o == n
for v in (t_ts, t_ck, t_ix): v.flush()
meta = dict(n=n, shard=a.shard, nshards=a.nshards, teacher_run=a.teacher, teacher_ckpt=ck_path, teacher_sha256=hashlib.sha256(open(ck_path, "rb").read()).hexdigest(),
            rate=1.0, root=a.root, augment_percentage=0.0, seconds=round(time.time() - t0))
json.dump(meta, open(pre + "meta.json", "w"), indent=1); print(json.dumps(meta)); print("TEACHER_CACHE_DONE")
