"""Why do the student (GlobalConfig + train args, plug-in weights) and the plug-in teacher (its saved config.json, same
weights) differ in eval mode? Same input tensors, fp32; then a config diff."""
import os, sys, json
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "team_code"))
import jsonpickle, jsonpickle.ext.numpy as jsonpickle_numpy; jsonpickle_numpy.register_handlers()
from config import GlobalConfig
from model import LidarCenterNet
import data as datamod
P = "/work/gn21/n21001/carla_garage_runs/tfpp_v2x_000"; ROOT = "/work/gn21/n21001/carla_garage_smoke_root"; dev = "cuda"
sd = torch.load(P + "/model_0030.pth", map_location="cpu")
cs = GlobalConfig(); cs.initialize(root_dir=[ROOT], setting="all", use_v2x=1, v2x_rate_dropout=1, v2x_p_zero=0.25, image_architecture="regnety_032",
                                   lidar_architecture="regnety_032", use_controller_input_prediction=1)
ct = GlobalConfig(); ct.__dict__.update(jsonpickle.decode(open(P + "/config.json").read()).__dict__)
diff = {k: (cs.__dict__.get(k), ct.__dict__.get(k)) for k in sorted(set(cs.__dict__) | set(ct.__dict__))
        if k not in ("data_roots", "root_dir", "detailed_loss_weights") and str(cs.__dict__.get(k)) != str(ct.__dict__.get(k))}
print(f"{len(diff)} config fields differ (student vs plug-in config.json):")
for k, (a, b) in diff.items(): print(f"  {k:40s} student={str(a)[:60]:60s} teacher={str(b)[:60]}")
cs.augment_percentage = 0.0; cs.use_color_aug = 0
ds = datamod.CARLA_Data(root=cs.data_roots, config=cs, estimate_class_distributions=False, estimate_sem_distribution=False, shared_dict=None, rank=0)
d = next(iter(torch.utils.data.DataLoader(torch.utils.data.Subset(ds, list(range(0, len(ds), max(1, len(ds) // 8)))[:8]), batch_size=8, num_workers=0)))
kw = dict(rgb=d["rgb"].to(dev, dtype=torch.float32), lidar_bev=d["lidar"].to(dev, dtype=torch.float32), target_point=d["target_point"].to(dev, dtype=torch.float32),
          ego_vel=d["speed"].to(dev, dtype=torch.float32).unsqueeze(1), command=d["command"].to(dev, dtype=torch.float32),
          coop_states=d["coop_states"].to(dev, dtype=torch.float32), coop_mask=d["coop_mask"].to(dev, dtype=torch.float32))
ns = LidarCenterNet(cs); ns.load_state_dict(sd, strict=True); ns.to(dev).eval()
nt = LidarCenterNet(ct); nt.load_state_dict(sd, strict=True); nt.to(dev).eval()
with torch.no_grad():
    os_, ot = ns(**kw), nt(**kw)
    print(f"same inputs, eval, fp32: max |Δ ts logits| {float((os_[1]-ot[1]).abs().max()):.2e}, max |Δ ckpt| {float((os_[2]-ot[2]).abs().max()):.2e} m")
    ct.use_semantic = ct.use_bev_semantic = ct.use_depth = ct.detect_boxes = 0; ot2 = nt(**kw)
    print(f"teacher with perception heads skipped: max |Δ ckpt| vs teacher-full {float((ot2[2]-ot[2]).abs().max()):.2e} m, vs student {float((os_[2]-ot2[2]).abs().max()):.2e} m")
    ot3 = nt(**{k: v[:5] for k, v in kw.items()}); print(f"teacher on a 5-frame subset vs its full-batch rows: max |Δ ckpt| {float((ot3[2]-ot2[2][:5]).abs().max()):.2e} m")
    ot4 = nt(**kw); print(f"teacher run twice, same batch: max |Δ ckpt| {float((ot4[2]-ot2[2]).abs().max()):.2e} m")
print("DEBUG_DONE")
