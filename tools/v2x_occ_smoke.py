"""Smoke test for scheme C (occlusion-weighted loss + hidden-hazard auxiliary head) -- run before any training.
  python tools/v2x_occ_smoke.py --root /work/gn21/n21001/carla_garage_smoke_root
Checks: (1) dataset emits coop_hidden / coop_hazard / slowdown with plausible rates; (2) weighted losses equal the unweighted
ones when every weight is 1 (numerical identity); (3) loss_hidden_hazard is present, finite, and zero-masked when rate = 0;
(4) gradients reach the auxiliary head; (5) config round-trips through jsonpickle with the new fields; (6) the closed-loop agent
path (forward without labels) still returns the 10-tuple.
"""
import argparse, os, sys, json
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "team_code"))
import jsonpickle, jsonpickle.ext.numpy as jsonpickle_numpy; jsonpickle_numpy.register_handlers()
import config as cfgmod, model as modmod, data as datamod, v2x_features

ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--n", type=int, default=640); a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"; ok = True
def check(cond, msg):
    global ok; print(("PASS " if cond else "FAIL ") + msg); ok = ok and bool(cond)

cfg = cfgmod.GlobalConfig(); cfg.initialize(root_dir=[a.root], setting="all", use_v2x=1, use_v2x_aux=1, use_v2x_aux_reg=1, v2x_occ_weight=4.0, v2x_rate_dropout=1, v2x_p_zero=0.1)
check("loss_hidden_hazard" in cfg.detailed_loss_weights and "loss_hidden_reg" in cfg.detailed_loss_weights, "config has loss_hidden_hazard / loss_hidden_reg weights")
ds = datamod.CARLA_Data(root=cfg.data_roots, config=cfg, estimate_class_distributions=False, estimate_sem_distribution=False, shared_dict=None, rank=0)
sub = torch.utils.data.Subset(ds, torch.randperm(len(ds), generator=torch.Generator().manual_seed(0))[: a.n].tolist())
dl = torch.utils.data.DataLoader(sub, batch_size=16, shuffle=False, num_workers=8)
# ---- (1) data statistics ----
occ_any, occ_slow, slow, hid_cnt, haz_cnt, n = 0, 0, 0, 0, 0, 0; first = None
for d in dl:
    hid, haz, mk, sl = d["coop_hidden"], d["coop_hazard"], d["coop_mask"], d["slowdown"].float()
    check(hid.shape == mk.shape and haz.shape == mk.shape and sl.shape == (mk.shape[0],), "shapes coop_hidden/coop_hazard (bs,K), slowdown (bs,)") if n == 0 else None
    oc = ((mk * hid * haz).sum(1) > 0).float(); occ_any += int(oc.sum()); occ_slow += int((oc * sl).sum()); slow += int(sl.sum())
    hid_cnt += int((mk * hid).sum()); haz_cnt += int((mk * haz).sum()); n += mk.shape[0]
    if first is None: first = d
print(f"frames {n}: hidden vehicles/frame {hid_cnt/n:.2f}, hazard vehicles/frame {haz_cnt/n:.2f}; frames with hidden hazard {100*occ_any/n:.1f}%, "
      f"expert slowing {100*slow/n:.1f}%, up-weighted (both) {100*occ_slow/n:.1f}%")
check(0 < occ_any < n, "hidden-hazard frames exist but are not all frames"); check(0 < slow < n, "slowdown label varies")
# ---- (2)(3)(4) losses ----
net = modmod.LidarCenterNet(cfg).to(dev); net.train()
check(hasattr(net, "hidden_hazard_head") and hasattr(net, "hidden_reg_head"), "aux heads built (hazard + reg)")
d = first; bs = d["rgb"].shape[0]
rgb = d["rgb"].to(dev, dtype=torch.float32); lidar = d["lidar"].to(dev, dtype=torch.float32); tp = d["target_point"].to(dev, dtype=torch.float32)
tpn = d["target_point_next"].to(dev, dtype=torch.float32) if cfg.two_tp_input else None; vel = d["speed"].to(dev, dtype=torch.float32).unsqueeze(1); cmd = d["command"].to(dev, dtype=torch.float32)
st = d["coop_states"].to(dev, dtype=torch.float32); mk = d["coop_mask"].to(dev, dtype=torch.float32); bk = d["coop_bucket"].to(dev)
ts = d["target_speed_twohot"].to(dev, dtype=torch.float32); ck = d["route"][:, : cfg.predict_checkpoint_len].to(dev, dtype=torch.float32)
labels = dict(semantic_label=d["semantic"].to(dev, dtype=torch.long) if cfg.use_semantic else None, bev_semantic_label=d["bev_semantic"].to(dev, dtype=torch.long) if cfg.use_bev_semantic else None,
              depth_label=d["depth"].to(dev, dtype=torch.float32) if cfg.use_depth else None, center_heatmap_label=d["center_heatmap"].to(dev, dtype=torch.float32), wh_label=d["wh"].to(dev, dtype=torch.float32),
              yaw_class_label=d["yaw_class"].to(dev, dtype=torch.long), yaw_res_label=d["yaw_res"].to(dev, dtype=torch.float32), offset_label=d["offset"].to(dev, dtype=torch.float32),
              velocity_label=d["velocity"].to(dev, dtype=torch.float32), brake_target_label=d["brake_target"].to(dev, dtype=torch.long), pixel_weight_label=d["pixel_weight"].to(dev, dtype=torch.float32),
              avg_factor_label=d["avg_factor"].to(dev, dtype=torch.float32))
def reg_targets(states, mask):
    cand = mask * d["coop_hidden"].to(dev) * d["coop_hazard"].to(dev); first = torch.argmax(cand, dim=1)
    return states[torch.arange(states.shape[0], device=dev), first, :4], (cand.sum(1) > 0).float()
def run(states, mask, weight, aux_label, aux_mask, reg=True):
    out = net(rgb=rgb, lidar_bev=lidar, target_point=tp, ego_vel=vel, command=cmd, target_point_next=tpn, coop_states=states, coop_mask=mask)
    rl, rm = reg_targets(states, mask) if reg else (None, None)
    return net.compute_loss(pred_wp=out[0], pred_target_speed=out[1], pred_checkpoint=out[2], pred_semantic=out[3], pred_bev_semantic=out[4], pred_depth=out[5],
                            pred_bounding_box=out[6], pred_wp_1=out[8], selected_path=out[9], waypoint_label=None, target_speed_label=ts, checkpoint_label=ck,
                            sample_weight=weight, aux_label=aux_label, aux_mask=aux_mask, aux_reg_label=rl, aux_reg_mask=rm, **labels), out
with torch.no_grad():
    net.eval()
    l_plain, _ = run(st, mk, None, None, None, reg=False)
    l_w1, _ = run(st, mk, torch.ones(bs, device=dev), None, None, reg=False)
    check(abs(float(l_plain["loss_target_speed"] - l_w1["loss_target_speed"])) < 1e-5 and abs(float(l_plain["loss_checkpoint"] - l_w1["loss_checkpoint"])) < 1e-5,
          f"weight==1 reproduces unweighted losses (ts {float(l_plain['loss_target_speed']):.4f} vs {float(l_w1['loss_target_speed']):.4f})")
    check("loss_hidden_hazard" not in l_plain, "no aux loss when no aux label passed")
    # scheme C path as in train.py
    s2, m2, rates = v2x_features.apply_random_rate(st, mk, bk, p_zero=0.5)
    occ = ((m2 * d["coop_hidden"].to(dev) * d["coop_hazard"].to(dev)).sum(1) > 0).float(); w = 1 + 3 * occ * d["slowdown"].to(dev).float(); w = w / w.mean()
    l_c, out = run(s2, m2, w, occ, (rates > 0).float())
    check("loss_hidden_hazard" in l_c and torch.isfinite(l_c["loss_hidden_hazard"]), f"aux loss present and finite: {float(l_c['loss_hidden_hazard']):.4f} (rate>0 samples {int((rates>0).sum())}/{bs})")
    rl, rm = reg_targets(s2, m2)
    check("loss_hidden_reg" in l_c and torch.isfinite(l_c["loss_hidden_reg"]), f"reg loss present and finite: {float(l_c['loss_hidden_reg']):.4f} (samples with a connected hidden hazard {int(rm.sum())}/{bs})")
    check(bool(((rl.abs().sum(1) > 0) | (rm == 0)).all()), "reg targets are non-zero exactly where a connected hidden hazard exists")
    check(all(torch.isfinite(v) for v in l_c.values()), "all losses finite: " + ", ".join(f"{k}={float(v):.3f}" for k, v in l_c.items()))
    l_z, _ = run(s2, torch.zeros_like(mk), w, torch.zeros(bs, device=dev), torch.zeros(bs, device=dev))
    check(float(l_z["loss_hidden_hazard"]) == 0.0, "aux loss is exactly 0 when every sample has rate 0 (mask all zero)")
    check(float(l_z["loss_hidden_reg"]) == 0.0, "reg loss is exactly 0 when every sample has rate 0")
    check(len(out) == 10, "forward still returns the 10-tuple used by sensor_agent")
net.train(); l_c, _ = run(s2, m2, w, occ, (rates > 0).float()); total = sum(cfg.detailed_loss_weights[k] * v for k, v in l_c.items()); total.backward()
g = [n_ for n_, p in net.named_parameters() if "hidden_hazard_head" in n_ and p.grad is not None and p.grad.abs().sum() > 0]
check(len(g) == 4, f"grad reached aux head params {len(g)}/4")
g2 = [n_ for n_, p in net.named_parameters() if "hidden_reg_head" in n_ and p.grad is not None and p.grad.abs().sum() > 0]; check(len(g2) == 4, f"grad reached reg head params {len(g2)}/4"); check(all(k in cfg.detailed_loss_weights for k in l_c), "every loss key has a weight in config (train loop would KeyError otherwise)")
# ---- (5) config round trip ----
c2 = jsonpickle.decode(jsonpickle.encode(cfg)); check(getattr(c2, "use_v2x_aux", None) == 1 and getattr(c2, "use_v2x_aux_reg", None) == 1 and getattr(c2, "v2x_occ_weight", None) == 4.0 and getattr(c2, "v2x_p_zero", None) == 0.1, "config.json round-trip keeps scheme-C fields")
# baseline model (use_v2x=0) must still build and run compute_loss without any aux kwargs
cfg0 = cfgmod.GlobalConfig(); cfg0.initialize(root_dir=[a.root], setting="all")
net0 = modmod.LidarCenterNet(cfg0).to(dev).eval()
with torch.no_grad():
    out0 = net0(rgb=rgb, lidar_bev=lidar, target_point=tp, ego_vel=vel, command=cmd, target_point_next=tpn)
    l0 = net0.compute_loss(pred_wp=out0[0], pred_target_speed=out0[1], pred_checkpoint=out0[2], pred_semantic=out0[3], pred_bev_semantic=out0[4], pred_depth=out0[5],
                           pred_bounding_box=out0[6], pred_wp_1=out0[8], selected_path=out0[9], waypoint_label=None, target_speed_label=ts, checkpoint_label=ck, **labels)
check(len(out0) == 10 and "loss_hidden_hazard" not in l0 and "loss_hidden_reg" not in l0, "baseline (use_v2x=0) path unaffected")
print("OCC SMOKE PASSED" if ok else "OCC SMOKE FAILED")
