"""Offline check of a V2XState-head checkpoint (use_v2xstate_head) against the base TF++ on recorded frames.
  python tools/v2xstate_offline.py --root /work/gn21/n21001/carla_garage_data_root --vs <run_dir> --base <run_dir> [--n 3000]
Un-augmented frames (closed-loop distribution), the same 3000 frames as bridgedrive_offline / v2x_offline_rate_sweep
(seed-0 permutation). Reports checkpoint ADE / FDE vs the expert route and target-speed accuracy for both models, per
command; and for the V2XState head the mechanism read-outs: mean gate per command (does the state gate follow the intent?),
mean attention, and how much the gate / checkpoints move when the command is swapped (left <-> right) on the same frames.
Offline numbers are not closed-loop gates: the head must at least approach the GRU's ADE before LOCAL runs routes.
"""
import argparse, os, sys, json, copy
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "team_code"))
import jsonpickle, jsonpickle.ext.numpy as jsonpickle_numpy; jsonpickle_numpy.register_handlers()
import config as cfgmod, model as modmod, data as datamod

ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--vs", required=True); ap.add_argument("--base", required=True)
ap.add_argument("--n", type=int, default=3000); ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--out", default="/work/gn21/n21001/V2XState_Real/tmp/v2xstate_offline.json"); a = ap.parse_args(); dev = "cuda"
CMD = ["left", "right", "straight", "lanefollow", "change_left", "change_right"]


def load(run_dir):
  cfg = cfgmod.GlobalConfig(); saved = jsonpickle.decode(open(os.path.join(run_dir, "config.json")).read()); cfg.__dict__.update(saved.__dict__)
  net = modmod.LidarCenterNet(cfg); ck = sorted(f for f in os.listdir(run_dir) if f.startswith("model_") and f.endswith(".pth"))[-1]
  net.load_state_dict(torch.load(os.path.join(run_dir, ck), map_location="cpu"), strict=True); net.to(dev).eval()
  print(f"loaded {run_dir}/{ck} (strict)", flush=True); return cfg, net


cfg_v, net_v = load(a.vs); cfg_b, net_b = load(a.base)
assert getattr(cfg_v, "use_v2xstate_head", 0), "--vs must be a V2XState-head checkpoint"
hd = net_v.v2xstate_head; print(f"head: basis_source {hd.basis_source}, K {hd.num_basis}, T {hd.basis_temperature}, grid {hd.height}x{hd.width}", flush=True)
cfg_b.initialize(root_dir=[a.root], setting="all")
cfg_ds = copy.deepcopy(cfg_b); cfg_ds.use_v2x = int(getattr(cfg_v, "use_v2x", 0))
cfg_ds.augment_percentage = 0.0; cfg_ds.use_color_aug = 0; cfg_ds.lidar_aug_prob = 0.0
ds = datamod.CARLA_Data(root=cfg_ds.data_roots, config=cfg_ds, estimate_class_distributions=False, estimate_sem_distribution=False, shared_dict=None, rank=0)
USE_V2X = bool(cfg_ds.use_v2x)
g = torch.Generator().manual_seed(0); idx = torch.randperm(len(ds), generator=g)[: a.n].tolist()
dl = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, idx), batch_size=a.bs, shuffle=False, num_workers=16)
P = cfg_b.predict_checkpoint_len
rec = {k: dict(ade=[], fde=[], acc=[]) for k in ("base", "vs")}; cmds = []; gates = []; att = []; d_gate = []; d_ck = []; perpt = {"base": [], "vs": []}
with torch.no_grad():
  for data in dl:
    rgb = data["rgb"].to(dev, dtype=torch.float32); lidar = data["lidar"].to(dev, dtype=torch.float32)
    tp = data["target_point"].to(dev, dtype=torch.float32); vel = data["speed"].to(dev, dtype=torch.float32).unsqueeze(1)
    cmd = data["command"].to(dev, dtype=torch.float32); route = data["route"][:, :P].to(dev, dtype=torch.float32)
    ts_label = data["target_speed_twohot"].to(dev, dtype=torch.float32).argmax(1)
    kw = dict(rgb=rgb, lidar_bev=lidar, target_point=tp, ego_vel=vel, command=cmd)
    kwv = dict(kw)
    if USE_V2X: kwv.update(coop_states=data["coop_states"].to(dev, dtype=torch.float32), coop_mask=data["coop_mask"].to(dev, dtype=torch.float32))
    ob = net_b(**kw); ov = net_v(**kwv); vs_out = net_v._v2xs
    for k, o in (("base", ob), ("vs", ov)):
      d = torch.linalg.norm(o[2].float() - route, dim=-1)
      rec[k]["ade"] += d.mean(1).tolist(); rec[k]["fde"] += d[:, -1].tolist(); rec[k]["acc"] += (o[1].argmax(1) == ts_label).tolist(); perpt[k].append(d.mean(0).cpu())
    cmds += cmd.argmax(1).tolist(); gates.append(vs_out["gate"].float().cpu()); att += vs_out["attention"].float().mean(dim=(1, 2)).tolist()
    sw = cmd.clone(); sw[:, [0, 1]] = cmd[:, [1, 0]]   # left <-> right; other commands unchanged
    kws = dict(kwv); kws["command"] = sw; os_ = net_v(**kws)
    d_gate += (net_v._v2xs["gate"].float() - vs_out["gate"].float()).abs().sum(1).tolist()
    d_ck += torch.linalg.norm(os_[2].float() - ov[2].float(), dim=-1).mean(1).tolist()
    if len(cmds) % (a.bs * 40) == 0: print(f"  {len(cmds)} frames", flush=True)
cmds = np.array(cmds); G = torch.cat(gates).numpy(); d_gate = np.array(d_gate); d_ck = np.array(d_ck)
res = dict(n=len(cmds), ade={k: float(np.mean(v["ade"])) for k, v in rec.items()}, fde={k: float(np.mean(v["fde"])) for k, v in rec.items()},
           ts_acc={k: float(np.mean(v["acc"])) for k, v in rec.items()}, attention_mean=float(np.mean(att)), per_command={})
print(f"\nframes {res['n']}")
for k in perpt: print(f"  per-checkpoint error (m) {k:4s}: " + " ".join(f"{v:.3f}" for v in torch.stack(perpt[k]).mean(0).tolist()))
print(f"  checkpoint ADE (m): base GRU {res['ade']['base']:.3f}, V2XState head {res['ade']['vs']:.3f};  FDE: base {res['fde']['base']:.3f}, V2XState {res['fde']['vs']:.3f}")
print(f"  target-speed accuracy: base {100 * res['ts_acc']['base']:.1f}, V2XState {100 * res['ts_acc']['vs']:.1f};  mean attention {res['attention_mean']:.3f}")
print("  per command: n | ADE base / V2XState | mean gate [near, left, forward, right] | swap left<->right: |d gate|_1, plan shift (m)")
for c in range(6):
  sel = cmds == c
  if not sel.any(): continue
  ab = float(np.mean(np.array(rec["base"]["ade"])[sel])); av = float(np.mean(np.array(rec["vs"]["ade"])[sel])); gm = G[sel].mean(0)
  res["per_command"][CMD[c]] = dict(n=int(sel.sum()), ade_base=ab, ade_vs=av, gate=gm.tolist(), d_gate_swap=float(d_gate[sel].mean()), d_ck_swap=float(d_ck[sel].mean()))
  print(f"    {CMD[c]:12s} {int(sel.sum()):5d} | {ab:.3f} / {av:.3f} | " + " ".join(f"{x:.2f}" for x in gm) + f" | {d_gate[sel].mean():.3f}, {d_ck[sel].mean():.3f}")
json.dump(res, open(a.out, "w"), indent=1); print("V2XSTATE_OFFLINE_DONE")
