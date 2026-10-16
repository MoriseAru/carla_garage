"""Offline check of a BridgeDrive (diffusion-bridge head) checkpoint against the base TF++ on recorded frames.
  python tools/bridgedrive_offline.py --root /work/gn21/n21001/carla_garage_data_root --bd <run_dir> --base <run_dir> [--n 3000] [--seeds 3]
Un-augmented frames (closed-loop distribution), the same 3000 frames as v2x_offline_rate_sweep (seed 0 permutation).
Reports, per model: checkpoint ADE / FDE (m) vs the expert route, target-speed accuracy, and for the BridgeDrive head:
anchor-classification accuracy (argmax == label-nearest anchor), oracle-anchor ADE (nearest anchor, no refinement),
ADE of the chosen anchor before refinement, and the sampling spread (ADE std over --seeds samplings of the same frame;
the bridge injects noise once at t = T). Offline numbers are not closed-loop gates; they are the base-reproduction
sanity check (the head must at least match the GRU's ADE on expert frames) before LOCAL runs the routes.
"""
import argparse, os, sys, json, math, copy
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "team_code"))
import jsonpickle, jsonpickle.ext.numpy as jsonpickle_numpy; jsonpickle_numpy.register_handlers()
import config as cfgmod, model as modmod, data as datamod

ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--bd", required=True); ap.add_argument("--base", required=True)
ap.add_argument("--n", type=int, default=3000); ap.add_argument("--bs", type=int, default=16); ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--out", default="/work/gn21/n21001/V2XState_Real/tmp/bridgedrive_offline.json")
ap.add_argument("--zero_last_y", type=int, default=-1, help="override the head's zero_last_y flag (-1: as in the checkpoint's config)")
a = ap.parse_args(); dev = "cuda"


def load(run_dir):
  cfg = cfgmod.GlobalConfig(); saved = jsonpickle.decode(open(os.path.join(run_dir, "config.json")).read()); cfg.__dict__.update(saved.__dict__)
  net = modmod.LidarCenterNet(cfg); ck = sorted(f for f in os.listdir(run_dir) if f.startswith("model_") and f.endswith(".pth"))[-1]
  r = net.load_state_dict(torch.load(os.path.join(run_dir, ck), map_location="cpu"), strict=True); net.to(dev).eval()
  print(f"loaded {run_dir}/{ck} (strict)", flush=True); return cfg, net


cfg_d, net_d = load(a.bd); cfg_b, net_b = load(a.base)
assert getattr(cfg_d, "use_bridgedrive", 0), "--bd must be a BridgeDrive checkpoint"
head = net_d.bridgedrive_head
if a.zero_last_y >= 0: head.zero_last_y = bool(a.zero_last_y); print(f"zero_last_y overridden -> {head.zero_last_y}", flush=True)
print(f"head: anchors {tuple(head.plan_anchor.shape)}, steps {head.step_num}, zero_last_y {head.zero_last_y}, swap_xy {head.diff_decoder[0].cross_bev_attention.swap_xy}; "
      f"anchor first-point distance {torch.linalg.norm(head.plan_anchor[:, 0], dim=-1).min():.2f}-{torch.linalg.norm(head.plan_anchor[:, 0], dim=-1).max():.2f} m; "
      f"stats mean x[0] {head.mean[0, 0, 0, 0]:.3f} std x[0] {head.std[0, 0, 0, 0]:.3f}", flush=True)
cfg_b.initialize(root_dir=[a.root], setting="all")
cfg_ds = copy.deepcopy(cfg_b); cfg_ds.use_v2x = int(getattr(cfg_d, "use_v2x", 0))   # tokens in the batch when the BridgeDrive model is the plug-in version
cfg_ds.augment_percentage = 0.0; cfg_ds.use_color_aug = 0; cfg_ds.lidar_aug_prob = 0.0
ds = datamod.CARLA_Data(root=cfg_ds.data_roots, config=cfg_ds, estimate_class_distributions=False, estimate_sem_distribution=False, shared_dict=None, rank=0)
USE_V2X = bool(cfg_ds.use_v2x); print(f"plug-in version: {USE_V2X} (bd rows = tokens at rate 1; extra row bd_r0 = all slots null)", flush=True)
g = torch.Generator().manual_seed(0); idx = torch.randperm(len(ds), generator=g)[: a.n].tolist()
dl = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, idx), batch_size=a.bs, shuffle=False, num_workers=16)
P = cfg_b.predict_checkpoint_len
acc = {"base": [], "bd": []}; ade = {"base": [], "bd": [], "bd_anchor": [], "oracle_anchor": []}; fde = {"base": [], "bd": []}; perpt = {"base": [], "bd": []}
if USE_V2X: acc["bd_r0"] = []; ade["bd_r0"] = []; fde["bd_r0"] = []; perpt["bd_r0"] = []
cls_acc = []; spread = []; n = 0
with torch.no_grad():
  for data in dl:
    rgb = data["rgb"].to(dev, dtype=torch.float32); lidar = data["lidar"].to(dev, dtype=torch.float32)
    tp = data["target_point"].to(dev, dtype=torch.float32); tpn = data["target_point_next"].to(dev, dtype=torch.float32) if cfg_b.two_tp_input else None
    vel = data["speed"].to(dev, dtype=torch.float32).unsqueeze(1); cmd = data["command"].to(dev, dtype=torch.float32)
    ts_label = data["target_speed_twohot"].to(dev, dtype=torch.float32).argmax(1); route = data["route"][:, :P].to(dev, dtype=torch.float32)
    kw = dict(rgb=rgb, lidar_bev=lidar, target_point=tp, ego_vel=vel, command=cmd, target_point_next=tpn)
    ob = net_b(**kw)
    kwd = dict(kw); kw0 = None
    if USE_V2X:
      kwd.update(coop_states=data["coop_states"].to(dev, dtype=torch.float32), coop_mask=data["coop_mask"].to(dev, dtype=torch.float32))
      kw0 = dict(kwd); kw0["coop_mask"] = torch.zeros_like(kwd["coop_mask"])
    samples = []
    for s_ in range(a.seeds):
      torch.manual_seed(1000 * s_ + n); od = net_d(**kwd); samples.append(od[2].float())
    if kw0 is not None:
      torch.manual_seed(n); o0 = net_d(**kw0); pc0 = o0[2].float(); acc["bd_r0"] += (o0[1].argmax(1) == ts_label).tolist()
      d0 = torch.linalg.norm(pc0 - route, dim=-1); ade["bd_r0"] += d0.mean(1).tolist(); fde["bd_r0"] += d0[:, -1].tolist(); perpt["bd_r0"].append(d0.mean(0).cpu())
    pc_b, pc_d = ob[2].float(), samples[0]
    acc["base"] += (ob[1].argmax(1) == ts_label).tolist(); acc["bd"] += (od[1].argmax(1) == ts_label).tolist()
    for k, pc in (("base", pc_b), ("bd", pc_d)):
      d = torch.linalg.norm(pc - route, dim=-1); ade[k] += d.mean(1).tolist(); fde[k] += d[:, -1].tolist(); perpt[k].append(d.mean(0).cpu())
    if a.seeds > 1:
      st = torch.stack(samples); spread += torch.linalg.norm(st - st.mean(0, keepdim=True), dim=-1).mean(-1).std(0).tolist()
    # anchor classification: run the cls stack once (same as inside forward_test)
    bs = rgb.shape[0]
    bev_map = net_d.backbone(rgb, lidar)[1]
    fused = net_d.change_channel(bev_map); fused = fused + net_d.encoder_pos_encoding(fused); fused = torch.flatten(fused, start_dim=2)
    extra = net_d.extra_sensor_encoder(torch.cat([net_d.velocity_normalization(vel), cmd], 1)) + net_d.extra_sensor_pos_embed.repeat(bs, 1)
    fused = torch.permute(torch.cat((fused, extra.unsqueeze(2)), 2), (0, 2, 1))
    if USE_V2X: fused = torch.cat((fused, net_d.coop_tokens(kwd["coop_states"], kwd["coop_mask"], bs)), dim=1)
    q = net_d.join(net_d.checkpoint_query.repeat(bs, 1, 1), fused)[:, :P]
    anc = head.plan_anchor[None].repeat(bs, 1, 1, 1)
    logits = head.run_decoder_cls(head.encode_points(anc), anc, bev_map, q)[-1]
    dist = torch.linalg.norm(route[:, None] - anc, dim=-1).mean(-1)          # (bs, K)
    gt_mode = dist.argmin(1); pred_mode = logits.argmax(1)
    cls_acc += (pred_mode == gt_mode).tolist(); ade["oracle_anchor"] += dist.min(1).values.tolist(); ade["bd_anchor"] += dist.gather(1, pred_mode[:, None]).squeeze(1).tolist()
    n += bs
    if n % (a.bs * 40) == 0: print(f"  {n} frames", flush=True)
res = dict(n=n, ts_acc={k: float(np.mean(v)) for k, v in acc.items()}, ade={k: float(np.mean(v)) for k, v in ade.items()}, fde={k: float(np.mean(v)) for k, v in fde.items()},
           cls_acc=float(np.mean(cls_acc)), spread=float(np.mean(spread)) if spread else None)
print(f"\nframes {n}")
for k in perpt: print(f"  per-checkpoint error (m) {k:4s}: " + " ".join(f"{v:.3f}" for v in torch.stack(perpt[k]).mean(0).tolist()))
print(f"  checkpoint ADE (m): base GRU {res['ade']['base']:.3f}, bridgedrive {res['ade']['bd']:.3f};  FDE: base {res['fde']['base']:.3f}, bridgedrive {res['fde']['bd']:.3f}"
      + (f";  plug-in at rate 0: ADE {res['ade']['bd_r0']:.3f} FDE {res['fde']['bd_r0']:.3f} ts acc {100 * res['ts_acc']['bd_r0']:.1f}" if USE_V2X else ""))
print(f"  anchors: classification accuracy {100 * res['cls_acc']:.1f} %; ADE of the chosen anchor before refinement {res['ade']['bd_anchor']:.3f}, of the label-nearest anchor {res['ade']['oracle_anchor']:.3f}")
print(f"  target-speed accuracy: base {100 * res['ts_acc']['base']:.1f}, bridgedrive {100 * res['ts_acc']['bd']:.1f}" + (f";  sampling spread (ADE std over {a.seeds} samplings) {res['spread']:.3f} m" if spread else ""))
json.dump(res, open(a.out, "w"), indent=1); print("BRIDGEDRIVE_OFFLINE_DONE")
