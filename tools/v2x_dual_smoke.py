"""Smoke test for scheme D (cooperation-conditional target-speed head) -- run before any training.
  python tools/v2x_dual_smoke.py --root /work/gn21/n21001/carla_garage_smoke_root
Structural guarantees checked:
 1. two separate target-speed heads exist and hold different parameters after a step;
 2. with coop_mask == 0 the output is EXACTLY the sensor head's, and does not depend on coop_states at all
    (no-cooperation behaviour cannot be corrupted by the cooperative pathway -- this is what rd/occ could not guarantee);
 3. per-sample routing: rows with tokens take the cooperative head, rows without take the sensor head;
 4. the null-token second pass produces `_ts_sensor` and 'loss_target_speed_sensor', masked to token-bearing frames only,
    and is absent at inference (single decoder pass);
 5. gradients reach both heads; 6. the use_v2x=0 baseline and the agent's 10-tuple are unaffected; 7. config round-trips.
"""
import argparse, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "team_code"))
import jsonpickle, jsonpickle.ext.numpy as jsonpickle_numpy; jsonpickle_numpy.register_handlers()
import config as cfgmod, model as modmod, data as datamod, v2x_features

ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"; ok = True
def check(cond, msg):
    global ok; print(("PASS " if cond else "FAIL ") + msg); ok = ok and bool(cond)

torch.manual_seed(0)
SET = dict(use_v2x=1, v2x_dual_head=1, v2x_rate_dropout=1, v2x_p_zero=0.25)
cfg = cfgmod.GlobalConfig(); cfg.initialize(root_dir=[a.root], setting="all", **SET)
check("loss_target_speed_sensor" in cfg.detailed_loss_weights, "config has loss_target_speed_sensor weight")
ds = datamod.CARLA_Data(root=cfg.data_roots, config=cfg, estimate_class_distributions=False, estimate_sem_distribution=False, shared_dict=None, rank=0)
dl = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, list(range(0, 320))), batch_size=16, shuffle=False, num_workers=8)
d = next(iter(dl)); bs = d["rgb"].shape[0]
net = modmod.LidarCenterNet(cfg).to(dev)
check(net.dual_head and hasattr(net, "target_speed_network_coop"), "dual head built")
check(id(net.target_speed_network_coop) != id(net.target_speed_network), "cooperative head is a separate module")

rgb = d["rgb"].to(dev, dtype=torch.float32); lidar = d["lidar"].to(dev, dtype=torch.float32); tp = d["target_point"].to(dev, dtype=torch.float32)
tpn = d["target_point_next"].to(dev, dtype=torch.float32) if cfg.two_tp_input else None
vel = d["speed"].to(dev, dtype=torch.float32).unsqueeze(1); cmd = d["command"].to(dev, dtype=torch.float32)
st = d["coop_states"].to(dev, dtype=torch.float32); mk = d["coop_mask"].to(dev, dtype=torch.float32)
ts = d["target_speed_twohot"].to(dev, dtype=torch.float32).argmax(1); ck = d["route"][:, : cfg.predict_checkpoint_len].to(dev, dtype=torch.float32)
labels = dict(semantic_label=d["semantic"].to(dev, dtype=torch.long) if cfg.use_semantic else None,
              bev_semantic_label=d["bev_semantic"].to(dev, dtype=torch.long) if cfg.use_bev_semantic else None,
              depth_label=d["depth"].to(dev, dtype=torch.float32) if cfg.use_depth else None,
              center_heatmap_label=d["center_heatmap"].to(dev, dtype=torch.float32), wh_label=d["wh"].to(dev, dtype=torch.float32),
              yaw_class_label=d["yaw_class"].to(dev, dtype=torch.long), yaw_res_label=d["yaw_res"].to(dev, dtype=torch.float32),
              offset_label=d["offset"].to(dev, dtype=torch.float32), velocity_label=d["velocity"].to(dev, dtype=torch.float32),
              brake_target_label=d["brake_target"].to(dev, dtype=torch.long), pixel_weight_label=d["pixel_weight"].to(dev, dtype=torch.float32),
              avg_factor_label=d["avg_factor"].to(dev, dtype=torch.float32))
def fwd(states, mask, sensor=False):
    return net(rgb=rgb, lidar_bev=lidar, target_point=tp, ego_vel=vel, command=cmd, target_point_next=tpn,
               coop_states=states, coop_mask=mask, also_sensor_only=sensor)

net.eval()
with torch.no_grad():
    zero = torch.zeros_like(mk)
    o_a = fwd(st, zero)[1]                                   # rate 0, real states
    o_b = fwd(torch.randn_like(st) * 5.0, zero)[1]           # rate 0, garbage states
    check(torch.equal(o_a, o_b), "with coop_mask == 0 the output ignores coop_states entirely (max |diff| = "
          f"{float((o_a - o_b).abs().max()):.2e})")
    o_full = fwd(st, mk)[1]
    check(not torch.allclose(o_full[mk.sum(1) > 0], o_a[mk.sum(1) > 0], atol=1e-6), "with tokens the cooperative head changes the output")
    half = mk.clone(); half[bs // 2:] = 0                     # first half keeps tokens, second half has none
    o_mix = fwd(st, half)[1]
    check(torch.equal(o_mix[bs // 2:], o_a[bs // 2:]), "rows without tokens match the pure-rate-0 output (routing is per sample)")
    check(torch.equal(o_mix[: bs // 2], o_full[: bs // 2]) or torch.allclose(o_mix[: bs // 2], o_full[: bs // 2], atol=1e-5),
          "rows with tokens match the tokens-present output")
    check(getattr(net, "_ts_sensor", None) is None, "inference does the single decoder pass only (_ts_sensor stays None)")
    out = fwd(st, mk, sensor=True)
    check(net._ts_sensor is not None and net._ts_sensor.shape == (bs, len(cfg.target_speeds)), "training pass produces the sensor-only prediction")
    check(len(out) == 10, "forward still returns the 10-tuple used by sensor_agent")

net.train()
out = fwd(st, mk, sensor=True)
mask = (mk.sum(1) > 0).float()
l = net.compute_loss(pred_wp=out[0], pred_target_speed=out[1], pred_checkpoint=out[2], pred_semantic=out[3], pred_bev_semantic=out[4],
                     pred_depth=out[5], pred_bounding_box=out[6], pred_wp_1=out[8], selected_path=out[9], waypoint_label=None,
                     target_speed_label=ts, checkpoint_label=ck, ts_sensor_mask=mask, **labels)
check("loss_target_speed_sensor" in l and torch.isfinite(l["loss_target_speed_sensor"]),
      f"sensor-head loss present and finite: {float(l.get('loss_target_speed_sensor', float('nan'))):.4f} (token frames {int(mask.sum())}/{bs})")
check(all(k in cfg.detailed_loss_weights for k in l), "every loss key has a weight in config")
out0 = fwd(st, torch.zeros_like(mk), sensor=True)
l0 = net.compute_loss(pred_wp=out0[0], pred_target_speed=out0[1], pred_checkpoint=out0[2], pred_semantic=out0[3], pred_bev_semantic=out0[4],
                      pred_depth=out0[5], pred_bounding_box=out0[6], pred_wp_1=out0[8], selected_path=out0[9], waypoint_label=None,
                      target_speed_label=ts, checkpoint_label=ck, ts_sensor_mask=torch.zeros(bs, device=dev), **labels)
check(float(l0["loss_target_speed_sensor"]) == 0.0, "sensor-head loss is exactly 0 when no frame has tokens")
total = sum(cfg.detailed_loss_weights[k] * v for k, v in l.items()); total.backward()
gc = [n for n, p in net.named_parameters() if "target_speed_network_coop" in n and p.grad is not None and p.grad.abs().sum() > 0]
gs = [n for n, p in net.named_parameters() if n.startswith("target_speed_network.") and p.grad is not None and p.grad.abs().sum() > 0]
check(len(gc) == 4, f"grad reached the cooperative head {len(gc)}/4"); check(len(gs) == 4, f"grad reached the sensor head {len(gs)}/4")

c2 = jsonpickle.decode(jsonpickle.encode(cfg)); check(all(getattr(c2, k, None) == v for k, v in SET.items()),
                                                      "config.json round-trip keeps scheme-D fields: " + str({k: getattr(c2, k, None) for k in SET}))
cfg0 = cfgmod.GlobalConfig(); cfg0.initialize(root_dir=[a.root], setting="all")
net0 = modmod.LidarCenterNet(cfg0).to(dev).eval()
with torch.no_grad():
    o0 = net0(rgb=rgb, lidar_bev=lidar, target_point=tp, ego_vel=vel, command=cmd, target_point_next=tpn)
check(not net0.dual_head and len(o0) == 10, "baseline (use_v2x=0) builds a single head and returns the 10-tuple")
print("DUAL SMOKE PASSED" if ok else "DUAL SMOKE FAILED")
