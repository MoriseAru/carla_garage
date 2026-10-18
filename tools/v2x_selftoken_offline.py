"""Offline check of self-tokens (V2X_SELF) on recorded frames, with the real detection head of the token model.
  python tools/v2x_selftoken_offline.py --root /work/gn21/n21001/carla_garage_data_root --v2x <run_dir> --base <run_dir> [--n 3000]
Un-augmented frames only (the closed-loop distribution). Three questions, in order:
 A) Is the detection head independent of the tokens? (bitwise: the same boxes with rate-1 tokens and with none.) If not,
    self-tokens would feed back into the detections they are built from.
 B) Do the decoded detections live in the token frame? Each detection (class car / emergency, after the agent's NMS) is
    matched to the ground-truth tokens of the frame: position error, heading error (and 180-degree flips), length error,
    precision, and recall of VISIBLE (> 5 lidar hits) / hidden vehicles inside the +-32 m detection range. A frame or sign
    bug shows up here as metre-scale errors or near-zero precision.
 C) Does filling the empty slots with the ego's own detections remove the rate-0 collapse on these frames?
    base | v2x r1 | v2x r0 | r0 + GT tokens of visible vehicles inside +-32 m (perfect-detection bound) |
    r0 + self-tokens with the speed of the matched GT vehicle (offline stand-in for tracking: an upper bound) |
    r0 + self-tokens with speed 0 (lower bound) | r0.5 compacted | r0.5 + self-tokens (GT speed) | r0.25 likewise.
    The closed-loop agent estimates speed by tracking, which lies between the two bounds; offline frames are single
    snapshots, so no tracker runs here, and detections are from the SAME frame (the agent uses the previous one).
Metrics vs expert labels: target-speed accuracy, checkpoint L1, KL(base || condition) on the target-speed head,
accuracy on frames with a connected hidden hazard (occ). Offline numbers are not closed-loop gates (2026-10 lesson).
"""
import argparse, os, sys, json, math
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "team_code"))
import jsonpickle, jsonpickle.ext.numpy as jsonpickle_numpy; jsonpickle_numpy.register_handlers()
import config as cfgmod, model as modmod, data as datamod, v2x_features as V, transfuser_utils as t_u

ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--v2x", required=True); ap.add_argument("--base", required=True)
ap.add_argument("--n", type=int, default=3000); ap.add_argument("--bs", type=int, default=16); ap.add_argument("--match", type=float, default=2.5)
ap.add_argument("--spd_noise", default="", help="comma list of sigma (m/s): extra conditions r0_self_n<sigma> = GT speed + N(0, sigma), clipped at 0 -- how accurate must the tracked speed be?")
ap.add_argument("--out", default="/work/gn21/n21001/V2XState_Real/tmp/v2x_selftoken_offline.json")
a = ap.parse_args()
dev = "cuda"


def load(run_dir):
    cfg = cfgmod.GlobalConfig(); saved = jsonpickle.decode(open(os.path.join(run_dir, "config.json")).read()); cfg.__dict__.update(saved.__dict__)
    net = modmod.LidarCenterNet(cfg); ck = sorted(f for f in os.listdir(run_dir) if f.startswith("model_") and f.endswith(".pth"))[-1]
    r = net.load_state_dict(torch.load(os.path.join(run_dir, ck), map_location="cpu"), strict=False); net.to(dev).eval()
    print(f"  load {ck}: missing {list(r.missing_keys)[:5]} ({len(r.missing_keys)}), unexpected {list(r.unexpected_keys)[:5]} ({len(r.unexpected_keys)})", flush=True)
    print(f"loaded {run_dir}/{ck}", flush=True); return cfg, net


cfg_v, net_v = load(a.v2x); cfg_b, net_b = load(a.base)
assert getattr(cfg_v, "use_v2x", 0) and cfg_v.detect_boxes, "needs a token model with the detection head"
print(f"detection head velocity output active: {not (cfg_v.lidar_seq_len == 1 and cfg_v.seq_len == 1)} "
      f"(lidar_seq_len={cfg_v.lidar_seq_len}, seq_len={cfg_v.seq_len}); NMS iou {cfg_v.iou_treshold_nms}; conf {cfg_v.bb_confidence_threshold}", flush=True)
cfg_v.initialize(root_dir=[a.root], setting="all", use_v2x=1, use_v2x_bev=0)
cfg_v.augment_percentage = 0.0; cfg_v.use_color_aug = 0; cfg_v.lidar_aug_prob = 0.0
ds = datamod.CARLA_Data(root=cfg_v.data_roots, config=cfg_v, estimate_class_distributions=False, estimate_sem_distribution=False, shared_dict=None, rank=0)
g = torch.Generator().manual_seed(0); idx = torch.randperm(len(ds), generator=g)[: a.n].tolist()   # same frames as v2x_offline_rate_sweep
dl = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, idx), batch_size=a.bs, shuffle=False, num_workers=16)
K = cfg_v.v2x_k
HALF = float(cfg_v.max_x)


def decode(feats, i):
    """Agent path for one sample: convert_features_to_bb_metric + the NMS the agent applies to the (1-model) ensemble."""
    boxes = net_v.convert_features_to_bb_metric(tuple(None if f is None else f[i:i + 1] for f in feats))
    boxes = t_u.non_maximum_suppression([boxes], cfg_v.iou_treshold_nms)
    return [b for b in boxes if int(round(float(b[7]))) in (0, 4)]


def gt_rows(st, mk):
    """Ground-truth tokens of one sample -> list of (x, y, yaw, speed, length, slot)."""
    out = []
    for j in range(K):
        if mk[j] < 0.5: continue
        r = st[j]
        out.append((float(r[0]) * 32, float(r[1]) * 32, math.atan2(float(r[5]), float(r[4])), math.hypot(float(r[2]), float(r[3])) * 10, float(r[6]) * 5, j))
    return out


def build(senders, selfs):
    """Agent merge: message tokens + self-tokens farther than --match m from every sender; nearest-K; returns (K,7),(K,)."""
    cands = [(math.hypot(s[0], s[1]), V._state(*s[:5])) for s in senders]
    for e in selfs:
        if any((e[0] - s[0]) ** 2 + (e[1] - s[1]) ** 2 <= a.match ** 2 for s in senders): continue
        cands.append((math.hypot(e[0], e[1]), V._state(e[0], e[1], e[2], max(0.0, e[3]), e[4])))
    cands.sort(key=lambda t: t[0])
    s = np.zeros((K, V.STATE_DIM), np.float32); m = np.zeros((K,), np.float32)
    for j, (_, v) in enumerate(cands[:K]): s[j], m[j] = v, 1.0
    return s, m


conds = ["base", "v2x_r1", "v2x_r0", "r0_vis32", "r0_self_gtspd", "r0_self_zero", "r0.5_c", "r0.5_self", "r0.25_c", "r0.25_self"]
SIG = [float(x) for x in a.spd_noise.replace(":", ",").split(",") if x]   # ":" also accepted (qsub -v splits on commas)
conds += [f"r0_self_n{sg:g}" for sg in SIG]
rng_spd = np.random.default_rng(0)
acc = {c: [] for c in conds}; l1 = {c: [] for c in conds}; kl = {c: [] for c in conds}; occ_all = []; nself = {c: [] for c in conds}
spd_rec = {c: [] for c in conds}; lab_spd = []; lead_all = []   # decoded target speed (closed-loop rule) per condition; expert label; lead-vehicle frames
TS = torch.tensor(cfg_v.target_speeds, dtype=torch.float32)
def decode_speed(logits):
  pr = torch.softmax(logits.float().cpu(), 1)
  return torch.where(pr[:, 0] > float(cfg_v.brake_uncertainty_threshold), torch.zeros(pr.shape[0]), (pr * TS).sum(1))
det = dict(n_det=0, matched=0, pos=[], yaw=[], flip=0, length=[], vis_gt=0, vis_hit=0, hid_gt=0, hid_hit=0, gt_spd_of_matched=[])
indep_max = 0.0; n = 0
with torch.no_grad():
    for data in dl:
        rgb = data["rgb"].to(dev, dtype=torch.float32); lidar = data["lidar"].to(dev, dtype=torch.float32)
        tp = data["target_point"].to(dev, dtype=torch.float32); tpn = data["target_point_next"].to(dev, dtype=torch.float32) if cfg_v.two_tp_input else None
        vel = data["speed"].to(dev, dtype=torch.float32).unsqueeze(1); cmd = data["command"].to(dev, dtype=torch.float32)
        ts_label = data["target_speed_twohot"].to(dev, dtype=torch.float32).argmax(1)
        ckpt_label = data["route"][:, : cfg_v.predict_checkpoint_len].to(dev, dtype=torch.float32)
        st = data["coop_states"].float(); mk = data["coop_mask"].float(); bk = data["coop_bucket"]; hid = data["coop_hidden"].float()
        occ = ((mk * hid * data["coop_hazard"].float()).sum(1) > 0); occ_all += occ.tolist(); bs = rgb.shape[0]
        lead = ((mk > 0.5) & (st[..., 0] * 32 > 0) & (st[..., 0] * 32 <= 20) & ((st[..., 1] * 32).abs() <= 2.5)).any(1); lead_all += lead.tolist()   # a car ahead in the ego lane within 20 m
        lab_spd += (data["target_speed_twohot"].float() * TS[None]).sum(1).tolist()
        kw = dict(rgb=rgb, lidar_bev=lidar, target_point=tp, ego_vel=vel, command=cmd, target_point_next=tpn)
        out_r1 = net_v(**kw, coop_states=st.to(dev), coop_mask=mk.to(dev))
        out_r0 = net_v(**kw, coop_states=st.to(dev), coop_mask=torch.zeros_like(mk).to(dev))
        # A) detection independence (index 6 = pred_bb_features)
        for f1, f0 in zip(out_r1[6], out_r0[6]):
            if f1 is None: continue          # velocity / brake heads are not run when lidar_seq_len = seq_len = 1
            indep_max = max(indep_max, float((f1.float() - f0.float()).abs().max()))
        feats = out_r1[6]
        S = {c: np.zeros((bs, K, V.STATE_DIM), np.float32) for c in conds}; M = {c: np.zeros((bs, K), np.float32) for c in conds}
        for i in range(bs):
            gts = gt_rows(st[i].numpy(), mk[i].numpy())
            dets = decode(feats, i)
            # B) detections vs ground truth
            selfs_gt, selfs_zero = [], []
            used = set()
            for b in dets:
                x, y, yaw, length = float(b[0]), float(b[1]), float(b[4]), 2.0 * float(b[2])
                det["n_det"] += 1
                d = [(math.hypot(gg[0] - x, gg[1] - y), gi) for gi, gg in enumerate(gts)]
                dmin, gi = min(d) if d else (1e9, -1)
                spd = 0.0
                if dmin <= a.match:
                    gg = gts[gi]; det["matched"] += 1; used.add(gi)
                    det["pos"].append(dmin); ye = abs(V.wrap_angle(yaw - gg[2])); det["yaw"].append(ye); det["flip"] += int(ye > math.pi / 2)
                    det["length"].append(length - gg[4]); spd = gg[3]; det["gt_spd_of_matched"].append(gg[3])
                selfs_gt.append((x, y, yaw, spd, length)); selfs_zero.append((x, y, yaw, 0.0, length))
            for gi, gg in enumerate(gts):
                if abs(gg[0]) < HALF and abs(gg[1]) < HALF:
                    key = "hid" if hid[i, gg[5]] > 0.5 else "vis"
                    det[key + "_gt"] += 1
                    det[key + "_hit"] += int(any(math.hypot(e[0] - gg[0], e[1] - gg[1]) <= a.match for e in selfs_gt))
            # C) token sets
            S["v2x_r1"][i], M["v2x_r1"][i] = st[i].numpy(), mk[i].numpy()
            vis32 = [gg for gg in gts if hid[i, gg[5]] < 0.5 and abs(gg[0]) < HALF and abs(gg[1]) < HALF]
            S["r0_vis32"][i], M["r0_vis32"][i] = build([], vis32)
            S["r0_self_gtspd"][i], M["r0_self_gtspd"][i] = build([], selfs_gt)
            S["r0_self_zero"][i], M["r0_self_zero"][i] = build([], selfs_zero)
            for sg in SIG:
                S[f"r0_self_n{sg:g}"][i], M[f"r0_self_n{sg:g}"][i] = build([], [(e[0], e[1], e[2], max(0.0, e[3] + float(rng_spd.normal(0.0, sg))), e[4]) for e in selfs_gt])
            for r in ("0.5", "0.25"):
                snd = [gg for gg in gts if int(bk[i, gg[5]]) < int(float(r) * V.BUCKETS)]
                S[f"r{r}_c"][i], M[f"r{r}_c"][i] = build(snd, [])
                S[f"r{r}_self"][i], M[f"r{r}_self"][i] = build(snd, selfs_gt)
        outs = {}
        for c in conds:
            if c == "base": o = net_b(**kw)
            elif c == "v2x_r1": o = out_r1
            elif c == "v2x_r0": o = out_r0
            else: o = net_v(**kw, coop_states=torch.from_numpy(S[c]).to(dev), coop_mask=torch.from_numpy(M[c]).to(dev))
            outs[c] = o; nself[c] += M[c].sum(1).tolist()
        lb = F.log_softmax(outs["base"][1].float(), 1)
        for c in conds:
            p = outs[c][1].float()
            acc[c] += (p.argmax(1) == ts_label).tolist(); spd_rec[c] += decode_speed(p).tolist()
            l1[c] += (outs[c][2].float() - ckpt_label).abs().mean(dim=(1, 2)).tolist()
            kl[c] += (lb.exp() * (lb - F.log_softmax(p, 1))).sum(1).tolist()
        n += bs
        if n % (a.bs * 40) == 0: print(f"  {n} frames", flush=True)

occ_all = np.array(occ_all, dtype=bool)
print(f"\nframes {n}; occ frames {int(occ_all.sum())}")
print(f"\nA) detection head vs tokens: max |pred_bb_features(r1) - pred_bb_features(r0)| = {indep_max:.3e} "
      f"-> {'INDEPENDENT' if indep_max == 0.0 else 'DEPENDS ON TOKENS'}")
med = lambda v, q=50: float(np.percentile(v, q)) if len(v) else float("nan")
print(f"\nB) detections (car/emergency, after NMS) vs ground-truth tokens, match radius {a.match} m:")
print(f"  {det['n_det']} detections, precision {det['matched']/max(det['n_det'],1):.3f}; position err median {med(det['pos']):.2f} p90 {med(det['pos'],90):.2f} m; "
      f"heading err median {math.degrees(med(det['yaw'])):.1f} deg, 180-flips {det['flip']} ({100*det['flip']/max(det['matched'],1):.1f}%); "
      f"length err median {med(det['length']):+.2f} m")
print(f"  recall inside +-{HALF:.0f} m: visible {det['vis_hit']/max(det['vis_gt'],1):.3f} (n={det['vis_gt']}), hidden {det['hid_hit']/max(det['hid_gt'],1):.3f} (n={det['hid_gt']})")
print(f"  matched vehicles moving > 1 m/s: {100*np.mean(np.array(det['gt_spd_of_matched'])>1.0):.0f}% (the share for which speed 0 is wrong)")
print(f"\nC) {'condition':15s} {'ts acc%':>8s} {'occ acc%':>9s} {'ckpt L1':>8s} {'KL(base||c)':>12s} {'tokens/frame':>13s}")
res = {}
for c in conds:
    A = np.array(acc[c], dtype=float)
    res[c] = dict(acc=A.mean(), occ=A[occ_all].mean() if occ_all.any() else float("nan"), l1=float(np.mean(l1[c])), kl=float(np.mean(kl[c])), tok=float(np.mean(nself[c])))
    print(f"   {c:15s} {100*res[c]['acc']:8.1f} {100*res[c]['occ']:9.1f} {res[c]['l1']:8.3f} {res[c]['kl']:12.4f} {res[c]['tok']:13.1f}")
lead_all = np.array(lead_all, dtype=bool); LS = np.array(lab_spd)
print(f"\nD) decoded target speed (closed-loop rule: P(stop) > {cfg_v.brake_uncertainty_threshold} -> 0, else expectation), m/s: mean / mean(pred - expert) / over-speed rate (pred > expert + 2 m/s)")
print(f"   expert label mean: all {LS.mean():.2f}, lead-vehicle frames (n={int(lead_all.sum())}) {LS[lead_all].mean():.2f}, occ frames (n={int(occ_all.sum())}) {LS[occ_all].mean():.2f}")
for c in conds:
  P = np.array(spd_rec[c]); row = []
  for nm, sel in (("all", np.ones_like(lead_all)), ("lead", lead_all), ("occ", occ_all)):
    d = P[sel] - LS[sel]; row.append(f"{nm} {P[sel].mean():5.2f} / {d.mean():+5.2f} / {100*np.mean(d > 2.0):4.1f}%")
    res[c][f"spd_{nm}"] = float(P[sel].mean()); res[c][f"dspd_{nm}"] = float(d.mean()); res[c][f"over_{nm}"] = float(np.mean(d > 2.0))
  print(f"   {c:15s} " + ";  ".join(row))
print(f"\n  R0 GAP (acc - base): v2x_r0 {100*(res['v2x_r0']['acc']-res['base']['acc']):+.1f} pt; r0_self_gtspd {100*(res['r0_self_gtspd']['acc']-res['base']['acc']):+.1f} pt; "
      f"r0_self_zero {100*(res['r0_self_zero']['acc']-res['base']['acc']):+.1f} pt; r0_vis32 {100*(res['r0_vis32']['acc']-res['base']['acc']):+.1f} pt")
json.dump(dict(n=n, indep_max=indep_max, det={k: (v if not isinstance(v, list) else dict(median=med(v), p90=med(v, 90), n=len(v))) for k, v in det.items()}, cond=res),
          open(a.out, "w"), indent=1)
print("SELFTOKEN_OFFLINE_DONE")
