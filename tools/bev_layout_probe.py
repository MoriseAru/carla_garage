"""Which (col, row) layout do the lidar input, the CenterNet label heatmap and the predicted heatmap use? Same cars (coop
tokens = GT car positions x forward / y right, metres) sampled under swap0 (col = x, row = y) and swap1 (col = y, row = x)."""
import os, sys, copy, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "team_code"))
import jsonpickle, jsonpickle.ext.numpy as jn; jn.register_handlers()
import config as cfgmod, model as modmod, data as datamod
base = "/work/gn21/n21001/carla_garage_runs/tfpp_base_000"
cfg = cfgmod.GlobalConfig(); cfg.__dict__.update(jsonpickle.decode(open(f"{base}/config.json").read()).__dict__)
cfg.initialize(root_dir=["/work/gn21/n21001/carla_garage_smoke_root"], setting="all")
cd = copy.deepcopy(cfg); cd.use_v2x = 1; cd.augment_percentage = 0.0; cd.use_color_aug = 0; cd.lidar_aug_prob = 0.0
ds = datamod.CARLA_Data(root=cd.data_roots, config=cd, estimate_class_distributions=False, estimate_sem_distribution=False, shared_dict=None, rank=0)
idx = torch.randperm(len(ds), generator=torch.Generator().manual_seed(0))[:8].tolist()
d = next(iter(torch.utils.data.DataLoader(torch.utils.data.Subset(ds, idx), batch_size=8, num_workers=4)))
net = modmod.LidarCenterNet(cfg); net.load_state_dict(torch.load(f"{base}/model_0030.pth", map_location="cpu"), strict=True); net.cuda().eval()
with torch.no_grad():
  o = net(rgb=d["rgb"].cuda().float(), lidar_bev=d["lidar"].cuda().float(), target_point=d["target_point"].cuda().float(),
          ego_vel=d["speed"].cuda().float()[:, None], command=d["command"].cuda().float())
maps = {"lidar input (256)": d["lidar"][:, 0].float(), "label heatmap car (64)": d["center_heatmap"][:, 0].float(),
        "pred heatmap car (64)": o[6][0][:, 0].cpu().float(), "bev semantic label vehicle (256)": (d["bev_semantic"] == 9).float() if "bev_semantic" in d else None}
print({k: tuple(v.shape) for k, v in maps.items() if v is not None})
cars = [(b, float(d["coop_states"][b, j, 0]) * 32, float(d["coop_states"][b, j, 1]) * 32) for b in range(8) for j in range(16)
        if d["coop_mask"][b, j] > 0.5 and d["coop_hidden"][b, j] < 0.5 and abs(float(d["coop_states"][b, j, 0]) * 32) < 30 and abs(float(d["coop_states"][b, j, 1]) * 32) < 30]
print(f"{len(cars)} cars")
for name, m in maps.items():
  if m is None: continue
  H = m.shape[-1]; px = H / 64.0
  def val(swap):
    v = []
    for b, x, y in cars:
      col, row = (y, x) if swap else (x, y)
      v.append(float(m[b, min(int((row + 32) * px), H - 1), min(int((col + 32) * px), H - 1)]))
    return np.mean(v)
  print(f"  {name:34s} swap0 (col=x,row=y) {val(0):.3f}   swap1 (col=y,row=x) {val(1):.3f}   mean {float(m.mean()):.4f}")
print("PROBE_DONE")
