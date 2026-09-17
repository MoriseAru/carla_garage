"""Cache the frozen base model's planning features for adapter training (scheme A).
  python tools/v2x_cache_features.py --root <data_root> --base <base_run_dir> --out <cache_dir> --shard 0 --nshards 2
Per frame (deterministic samples: augment_percentage=0, no colour aug): decoder output for the checkpoint queries
(joined, (L+1, d) fp16), decoder memory (S, d) fp16, the base model's own predictions, labels, and the coop tensors.
The base is run once; every adapter variant then trains on this cache in minutes.
"""
import argparse, os, sys, json, time, hashlib, zlib
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "team_code"))
import jsonpickle, jsonpickle.ext.numpy as jsonpickle_numpy; jsonpickle_numpy.register_handlers()
import config as cfgmod, model as modmod, data as datamod

ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--base", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1); ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--workers", type=int, default=48); ap.add_argument("--max_frames", type=int, default=0); a = ap.parse_args(); dev = "cuda"

def base_cfg():
    cfg = cfgmod.GlobalConfig(); saved = jsonpickle.decode(open(os.path.join(a.base, "config.json")).read()); cfg.__dict__.update(saved.__dict__); return cfg
cfg_d = base_cfg(); cfg_d.initialize(root_dir=[a.root], setting="all", use_v2x=1)
cfg_d.augment_percentage = 0.0; cfg_d.use_color_aug = 0; cfg_d.lidar_aug_prob = 0.0
ds = datamod.CARLA_Data(root=cfg_d.data_roots, config=cfg_d, estimate_class_distributions=False, estimate_sem_distribution=False, shared_dict=None, rank=0)
idx = np.array_split(np.arange(len(ds)), a.nshards)[a.shard]
if a.max_frames: idx = idx[: a.max_frames]
n = len(idx); L = cfg_d.predict_checkpoint_len; K = cfg_d.v2x_k; D = cfg_d.gru_input_size
print(f"dataset {len(ds)} frames; shard {a.shard}/{a.nshards}: {n} frames [{idx[0]}..{idx[-1]}]; L={L} K={K} d={D}", flush=True)

cfg_m = base_cfg(); assert not cfg_m.use_v2x, "base run must be a plain TF++ model"
net = modmod.LidarCenterNet(cfg_m); ck = sorted(f for f in os.listdir(a.base) if f.startswith("model_") and f.endswith(".pth"))[-1]
ck_path = os.path.join(a.base, ck); net.load_state_dict(torch.load(ck_path, map_location="cpu"), strict=True); net.to(dev).eval()
cfg_m.use_semantic = cfg_m.use_bev_semantic = cfg_m.use_depth = cfg_m.detect_boxes = 0   # skip the perception heads in forward (weights already loaded)
store = {}
net.join.register_forward_hook(lambda m, inp, out: store.update(memory=inp[1].detach(), joined=out.detach()))

os.makedirs(a.out, exist_ok=True); pre = os.path.join(a.out, f"shard{a.shard:02d}_")
def mm(name, shape, dtype): return np.lib.format.open_memmap(pre + name + ".npy", mode="w+", dtype=dtype, shape=shape)
arr = dict(joined=mm("joined", (n, L + 1, D), np.float16), base_ts=mm("base_ts", (n, len(cfg_d.target_speeds)), np.float32), base_ckpt=mm("base_ckpt", (n, L, 2), np.float32),
           target_point=mm("target_point", (n, 2), np.float32), speed=mm("speed", (n,), np.float32), command=mm("command", (n, 6), np.float32),
           ts_twohot=mm("ts_twohot", (n, len(cfg_d.target_speeds)), np.float32), route=mm("route", (n, L, 2), np.float32),
           coop_states=mm("coop_states", (n, K, cfg_d.v2x_state_dim), np.float32), coop_mask=mm("coop_mask", (n, K), np.uint8), coop_bucket=mm("coop_bucket", (n, K), np.int16),
           coop_hidden=mm("coop_hidden", (n, K), np.uint8), coop_hazard=mm("coop_hazard", (n, K), np.uint8), slowdown=mm("slowdown", (n,), np.uint8), brake=mm("brake", (n,), np.uint8),
           index=mm("index", (n,), np.int64))
memory = None; route_dirs = []
dl = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, idx.tolist()), batch_size=a.bs, shuffle=False, num_workers=a.workers, pin_memory=True)
t0 = time.time(); o = 0
with torch.no_grad():
    for bi, data in enumerate(dl):
        rgb = data["rgb"].to(dev, dtype=torch.float32, non_blocking=True); lidar = data["lidar"].to(dev, dtype=torch.float32, non_blocking=True)
        tp = data["target_point"].to(dev, dtype=torch.float32); vel = data["speed"].to(dev, dtype=torch.float32).unsqueeze(1); cmd = data["command"].to(dev, dtype=torch.float32)
        out = net(rgb=rgb, lidar_bev=lidar, target_point=tp, ego_vel=vel, command=cmd)
        bs = rgb.shape[0]; sl = slice(o, o + bs)
        if memory is None:
            S = store["memory"].shape[1]; memory = mm("memory", (n, S, D), np.float16); print(f"memory tokens S={S}; joined {tuple(store['joined'].shape[1:])}", flush=True)
        arr["joined"][sl] = store["joined"].half().cpu().numpy(); memory[sl] = store["memory"].half().cpu().numpy()
        arr["base_ts"][sl] = out[1].float().cpu().numpy(); arr["base_ckpt"][sl] = out[2].float().cpu().numpy()
        arr["target_point"][sl] = data["target_point"].numpy(); arr["speed"][sl] = data["speed"].numpy(); arr["command"][sl] = data["command"].numpy()
        arr["ts_twohot"][sl] = data["target_speed_twohot"].numpy(); arr["route"][sl] = data["route"][:, :L].numpy()
        arr["coop_states"][sl] = data["coop_states"].numpy(); arr["coop_mask"][sl] = data["coop_mask"].numpy().astype(np.uint8); arr["coop_bucket"][sl] = data["coop_bucket"].numpy().astype(np.int16)
        arr["coop_hidden"][sl] = data["coop_hidden"].numpy().astype(np.uint8); arr["coop_hazard"][sl] = data["coop_hazard"].numpy().astype(np.uint8)
        arr["slowdown"][sl] = data["slowdown"].numpy().astype(np.uint8); arr["brake"][sl] = np.asarray(data["brake"]).astype(np.uint8).reshape(-1)
        arr["index"][sl] = idx[o:o + bs]
        for i in idx[o:o + bs]: route_dirs.append(os.path.dirname(os.path.dirname(str(ds.images[i][0], encoding="utf-8"))))
        o += bs
        if bi % 100 == 0 or bi == 10: print(f"  {o}/{n} frames, {o / (time.time() - t0):.1f} frames/s", flush=True)
assert o == n, (o, n)
for v in arr.values(): v.flush()
memory.flush(); np.save(pre + "route_dir.npy", np.array(route_dirs))
meta = dict(n=n, S=int(memory.shape[1]), L=L, K=K, d=D, shard=a.shard, nshards=a.nshards, base_run=a.base, base_ckpt=ck_path,
            base_sha256=hashlib.sha256(open(ck_path, "rb").read()).hexdigest(), root=a.root, augment_percentage=0.0, use_color_aug=0, lidar_aug_prob=0.0,
            fields=sorted(list(arr) + ["memory", "route_dir"]), seconds=round(time.time() - t0), coop_fill=float(arr["coop_mask"][:].sum(1).mean()))
json.dump(meta, open(pre + "meta.json", "w"), indent=1); print(json.dumps(meta)); print("CACHE_DONE")
