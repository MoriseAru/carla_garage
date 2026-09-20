"""Can the agent hold the plug-in and the V2X-free base model at the same time, and does the router switch between two
genuinely different policies?  Replicates sensor_agent.setup()'s loading path without CARLA.
  python tools/v2x_router_load_smoke.py --v2x <plugin_run_dir> --base <base_run_dir> --root <data_root>
"""
import argparse, os, sys, json
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "team_code"))
import jsonpickle, jsonpickle.ext.numpy as jsonpickle_numpy; jsonpickle_numpy.register_handlers()
from config import GlobalConfig
import model as modmod, data as datamod, v2x_features as V

ap = argparse.ArgumentParser(); ap.add_argument("--v2x", required=True); ap.add_argument("--base", required=True)
ap.add_argument("--root", required=True); a = ap.parse_args(); dev = "cuda"
def ok(c, m):
    print(("PASS " if c else "FAIL ") + m, flush=True)
    if not c: sys.exit("ROUTER LOAD SMOKE FAILED")

def load(run_dir):                       # exactly the agent's path: json -> GlobalConfig -> strict load
    with open(os.path.join(run_dir, "config.json"), "rt", encoding="utf-8") as f:
        cfg = GlobalConfig(); cfg.__dict__.update(jsonpickle.decode(f.read()).__dict__)
    ck = [f for f in sorted(os.listdir(run_dir)) if f.endswith(".pth") and f.startswith("model")]
    if len(ck) > 1:   # miyabi run dirs keep intermediate epochs; LOCAL's ~/runs/<id>/ must hold exactly one (the agent ensembles a dir)
        print(f"NOTE {run_dir} holds {len(ck)} model_*.pth ({', '.join(ck)}); taking the last. On LOCAL the ckpt dir must hold exactly one.")
    ck = ck[-1:]
    net = modmod.LidarCenterNet(cfg)
    if cfg.sync_batch_norm: net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net.load_state_dict(torch.load(os.path.join(run_dir, ck[0]), map_location="cpu"), strict=True)
    return cfg, net.to(dev).eval(), ck[0]

cfg_v, net_v, ck_v = load(a.v2x); cfg_b, net_b, ck_b = load(a.base)
ok(True, f"strict load under this fork: plug-in {os.path.basename(a.v2x)}/{ck_v}, base {os.path.basename(a.base)}/{ck_b}")
ok(getattr(cfg_v, "use_v2x", 0) == 1, f"plug-in config has use_v2x=1 (adapter={getattr(cfg_v,'use_v2x_adapter',0)}, dual={getattr(cfg_v,'v2x_dual_head',0)})")
ok(getattr(cfg_b, "use_v2x", 0) == 0, "base config has use_v2x=0 -- the router's fallback is genuinely V2X-free")
for k in ("backbone", "image_architecture", "lidar_architecture", "predict_checkpoint_len", "target_speeds", "use_controller_input_prediction"):
    ok(getattr(cfg_v, k) == getattr(cfg_b, k), f"shared architecture field {k} = {getattr(cfg_v, k)}")
mem = torch.cuda.memory_allocated() / 2 ** 30
ok(mem < 3.0, f"both models resident on one GPU: {mem:.2f} GiB allocated (CARLA needs ~6 GiB of the 20 GiB card)")

cfg_v.initialize(root_dir=[a.root], setting="all", use_v2x=1)
ds = datamod.CARLA_Data(root=cfg_v.data_roots, config=cfg_v, estimate_class_distributions=False, estimate_sem_distribution=False, shared_dict=None, rank=0)
dl = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, list(range(0, len(ds), max(1, len(ds) // 16)))[:16]), batch_size=8, shuffle=False, num_workers=0)
d = next(iter(dl))
kw = dict(rgb=d["rgb"].to(dev, dtype=torch.float32), lidar_bev=d["lidar"].to(dev, dtype=torch.float32),
          target_point=d["target_point"].to(dev, dtype=torch.float32), ego_vel=d["speed"].to(dev, dtype=torch.float32).unsqueeze(1),
          command=d["command"].to(dev, dtype=torch.float32))
st = d["coop_states"].to(dev, dtype=torch.float32); mk = d["coop_mask"].to(dev, dtype=torch.float32)
with torch.no_grad():
    o_v = net_v(**kw, coop_states=st, coop_mask=mk)
    o_v0 = net_v(**kw, coop_states=st, coop_mask=torch.zeros_like(mk))
    o_b = net_b(**kw)
    o_b_kw = net_b(**kw, coop_states=st, coop_mask=mk)    # the agent passes the kwargs to whichever net it picked
for name, o in (("plug-in r1", o_v), ("plug-in r0", o_v0), ("base", o_b)):
    ok(torch.isfinite(o[1]).all() and torch.isfinite(o[2]).all(), f"{name}: finite target-speed logits and checkpoints")
ok(torch.equal(o_b[1], o_b_kw[1]) and torch.equal(o_b[2], o_b_kw[2]), "base model ignores coop kwargs bitwise (safe for the agent to pass them unconditionally)")
dts = float((o_v[1].softmax(1) - o_b[1].softmax(1)).abs().mean()); dck = float((o_v[2] - o_b[2]).abs().mean())
ok(dts > 1e-3, f"router switches between two different policies: mean |Δp(target speed)| {dts:.4f}, mean |Δcheckpoint| {dck:.4f} m")
agree = float((o_v[1].argmax(1) == o_b[1].argmax(1)).float().mean())
print(f"target-speed argmax agreement plug-in vs base on this batch: {100*agree:.0f}%")
print("ROUTER LOAD SMOKE PASSED")
