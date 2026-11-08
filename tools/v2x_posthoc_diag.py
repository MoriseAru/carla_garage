"""Offline diagnostics of "post-hoc tokens are not read" (HANDOVER 6717, D2 / D4 / D5 / D6 / D8). No model is trained;
every number comes from forward passes of existing checkpoints on the same frames (seeded subset, no augmentation).
  python tools/v2x_posthoc_diag.py --root <data_root> --model <run_dir> --base <run_dir> [--n 8000] [--out x.json]
--model may be any token model (co-train plug-in, frozen-base adapter, late-fusion raster model); --base is the V2X-free
TF++ base. Each run_dir holds config.json and model_*.pth (the last one is loaded).

D2 counterfactual probe: model with its own tokens (tok) vs no tokens (none: every slot masked), another frame's tokens
   (shuf: batch roll), randomly rotated tokens (rnd). Target-speed argmax change rate; expected speed E[v] = sum p * v;
   share of frames with |dE| > 1 m/s; among them the share where tok is SLOWER. All frames and occluded-hazard frames.
D4 base residual: base CE / accuracy on occluded vs other frames; held-out (route-grouped) multinomial logistic probe of
   the expert target-speed class from the base planner feature (A) vs A + the raw token states (B) -> unique linear
   information of the tokens beyond what the base already encodes.
D5 counterfactual ego inputs: ego speed or LiDAR swapped with another frame's (batch roll: in-distribution values, wrong
   frame) for base and model; loss increase and the content gap (tok - rnd) under each swap.
D6 feature alignment: L2 norms of BEV memory tokens / speed token / valid coop tokens / null token; share of the (centred)
   coop-token energy inside the top-k PCA subspace of the BEV memory tokens (k = 90 % BEV variance) vs the k/d chance level.
D8 linear probes (route-grouped 5-fold): from the target-speed query feature of the planning decoder, decode ego speed
   (R2), hidden-hazard presence (AUC) and the nearest hidden hazard's position (R2); for base, model-tok and model-none.
"""
import argparse, os, sys, json, math
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "team_code"))
import jsonpickle, jsonpickle.ext.numpy as jsonpickle_numpy; jsonpickle_numpy.register_handlers()
import config as cfgmod, model as modmod, data as datamod, v2x_features

ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--model", required=True)
ap.add_argument("--base", required=True); ap.add_argument("--n", type=int, default=8000); ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--out", default=""); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args(); dev = "cuda"
torch.manual_seed(a.seed)


def load(run_dir):
    cfg = cfgmod.GlobalConfig(); saved = jsonpickle.decode(open(os.path.join(run_dir, "config.json")).read()); cfg.__dict__.update(saved.__dict__)
    net = modmod.LidarCenterNet(cfg); ck = sorted(f for f in os.listdir(run_dir) if f.startswith("model_") and f.endswith(".pth"))[-1]
    net.load_state_dict(torch.load(os.path.join(run_dir, ck), map_location="cpu"), strict=False); net.to(dev).eval()
    print(f"loaded {run_dir}/{ck}", flush=True); return cfg, net


cfg_m, net_m = load(a.model); cfg_b, net_b = load(a.base)
is_bev = bool(getattr(cfg_m, "use_v2x_bev", 0)); is_ad = bool(getattr(net_m, "v2x_adapter_on", False))
kind = "raster" if is_bev else ("adapter" if is_ad else "cotrain")
cfg_m.initialize(root_dir=[a.root], setting="all", use_v2x=0 if is_bev else 1, use_v2x_bev=int(is_bev))
cfg_m.augment_percentage = 0.0; cfg_m.use_color_aug = 0; cfg_m.lidar_aug_prob = 0.0
ds = datamod.CARLA_Data(root=cfg_m.data_roots, config=cfg_m, estimate_class_distributions=False, estimate_sem_distribution=False, shared_dict=None, rank=0)
g = torch.Generator().manual_seed(a.seed); idx = torch.randperm(len(ds), generator=g)[: a.n].tolist()
dl = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, idx), batch_size=a.bs, shuffle=False, num_workers=16)
route_of = [os.path.dirname(os.path.dirname(ds.images[i][0].decode() if isinstance(ds.images[i][0], bytes) else str(ds.images[i][0]))) for i in idx]
print(f"model {a.model} ({kind}); frames {len(idx)} of {len(ds)} from {len(set(route_of))} routes", flush=True)
speeds_v = torch.tensor(cfg_m.target_speeds, dtype=torch.float32, device=dev)

# ---------------- hooks: planner feature (target-speed query output), decoder memory, adapter tokens ----------------
cap = {}
def hook_ts(name):
    def f(mod, args): cap[name + "_feat"] = args[0].detach().float()
    return f
net_m.target_speed_network.register_forward_pre_hook(hook_ts("m")); net_b.target_speed_network.register_forward_pre_hook(hook_ts("b"))
def hook_mem(mod, args): cap["mem"] = args[1].detach().float()
net_m.join.layers[0].register_forward_pre_hook(hook_mem)
if is_ad:
    _orig_tokens = net_m.v2x_adapter.tokens
    def _tokens(s, m):
        out = _orig_tokens(s, m); cap["adtok"] = out[0].detach().float(); return out
    net_m.v2x_adapter.tokens = _tokens
g_rot = torch.Generator().manual_seed(1234)


def run_m(rgb, lidar, tp, vel, cmd, tpn, s, m):
    if is_bev:
        ras = v2x_features.rasterize_states(s, m, cfg_m.min_x, cfg_m.max_x, cfg_m.min_y, cfg_m.max_y, cfg_m.pixels_per_meter, getattr(cfg_m, "v2x_bev_width", 2.0))
        out = net_m(rgb=rgb, lidar_bev=torch.cat((lidar, ras), 1), target_point=tp, ego_vel=vel, command=cmd, target_point_next=tpn)
    else:
        out = net_m(rgb=rgb, lidar_bev=lidar, target_point=tp, ego_vel=vel, command=cmd, target_point_next=tpn, coop_states=s, coop_mask=m)
    return out[1].float()


def run_b(rgb, lidar, tp, vel, cmd, tpn):
    return net_b(rgb=rgb, lidar_bev=lidar, target_point=tp, ego_vel=vel, command=cmd, target_point_next=tpn)[1].float()


M_CONDS = ["tok", "none", "shuf", "rnd", "tok_spd", "rnd_spd", "tok_lid", "rnd_lid"]
B_CONDS = ["base", "base_spd", "base_lid"]
logits = {c: [] for c in M_CONDS + B_CONDS}
rec = {k: [] for k in ("label", "occ", "speed", "haz_xy", "states", "mask")}
feats = {k: [] for k in ("b", "m_tok", "m_none")}
d6 = {"bev_norm": [], "spd_norm": [], "coop_norm": [], "null_norm": [], "bev_sample": [], "coop_sample": []}
gs = torch.Generator().manual_seed(7)

with torch.no_grad():
    for data in dl:
        rgb = data["rgb"].to(dev, dtype=torch.float32); lidar = data["lidar"].to(dev, dtype=torch.float32)
        tp = data["target_point"].to(dev, dtype=torch.float32); tpn = data["target_point_next"].to(dev, dtype=torch.float32) if cfg_m.two_tp_input else None
        vel = data["speed"].to(dev, dtype=torch.float32).unsqueeze(1); cmd = data["command"].to(dev, dtype=torch.float32)
        lab = data["target_speed_twohot"].to(dev, dtype=torch.float32).argmax(1)
        st = data["coop_states"].to(dev, dtype=torch.float32); mk = data["coop_mask"].to(dev, dtype=torch.float32)
        hid = data["coop_hidden"].to(dev, dtype=torch.float32); haz = data["coop_hazard"].to(dev, dtype=torch.float32)
        hh = mk * hid * haz; occ = hh.sum(1) > 0; bs = rgb.shape[0]
        # nearest connected hidden hazard (slots are distance-sorted: first flagged slot); NaN where none
        first = torch.argmax((hh > 0).to(torch.int64), dim=1)
        xy = st[torch.arange(bs, device=dev), first, :2] * 32.0; xy[~occ] = float("nan")
        roll = lambda x: torch.roll(x, 1, dims=0)
        rnd_s = v2x_features.rotate_states(st, (torch.rand(mk.shape, generator=g_rot) * 2 * math.pi).to(dev))
        # base
        logits["base"].append(run_b(rgb, lidar, tp, vel, cmd, tpn)); feats["b"].append(cap["b_feat"])
        logits["base_spd"].append(run_b(rgb, lidar, tp, roll(vel), cmd, tpn))
        logits["base_lid"].append(run_b(rgb, roll(lidar), tp, vel, cmd, tpn))
        # model
        logits["tok"].append(run_m(rgb, lidar, tp, vel, cmd, tpn, st, mk)); feats["m_tok"].append(cap["m_feat"])
        if not is_bev:   # D6 on the clean tok pass
            mem = cap["mem"]; nb = 64
            d6["bev_norm"] += mem[:, :nb].norm(dim=-1).mean(1).tolist(); d6["spd_norm"] += mem[:, nb].norm(dim=-1).tolist()
            ct = mem[:, -cfg_m.v2x_k:] if kind == "cotrain" else cap["adtok"][:, -cfg_m.v2x_k:]
            v = mk > 0.5
            if v.any(): d6["coop_norm"] += ct[v].norm(dim=-1).tolist()
            if kind == "cotrain" and (~v).any(): d6["null_norm"] += ct[~v].norm(dim=-1)[:1].tolist()
            pick = torch.randint(0, nb, (bs, 4), generator=gs).to(dev)
            d6["bev_sample"].append(torch.gather(mem[:, :nb], 1, pick[..., None].expand(-1, -1, mem.shape[-1])).reshape(-1, mem.shape[-1]).cpu())
            if v.any(): d6["coop_sample"].append(ct[v][: 4 * bs].cpu())
        logits["none"].append(run_m(rgb, lidar, tp, vel, cmd, tpn, st, torch.zeros_like(mk))); feats["m_none"].append(cap["m_feat"])
        logits["shuf"].append(run_m(rgb, lidar, tp, vel, cmd, tpn, roll(st), roll(mk)))
        logits["rnd"].append(run_m(rgb, lidar, tp, vel, cmd, tpn, rnd_s, mk))
        logits["tok_spd"].append(run_m(rgb, lidar, tp, roll(vel), cmd, tpn, st, mk))
        logits["rnd_spd"].append(run_m(rgb, lidar, tp, roll(vel), cmd, tpn, rnd_s, mk))
        logits["tok_lid"].append(run_m(rgb, roll(lidar), tp, vel, cmd, tpn, st, mk))
        logits["rnd_lid"].append(run_m(rgb, roll(lidar), tp, vel, cmd, tpn, rnd_s, mk))
        rec["label"].append(lab); rec["occ"].append(occ); rec["speed"].append(vel[:, 0]); rec["haz_xy"].append(xy)
        rec["states"].append((st * mk[..., None]).reshape(bs, -1)); rec["mask"].append(mk)

L = {k: torch.cat(v).cpu() for k, v in logits.items()}
R = {k: torch.cat(v).cpu() for k, v in rec.items()}
FT = {k: torch.cat(v).cpu().numpy() for k, v in feats.items()}
lab, occ = R["label"], R["occ"].numpy(); N = len(lab); groups = np.array(route_of)
res = {"model": a.model, "base": a.base, "kind": kind, "n": N, "n_occ": int(occ.sum()), "routes": len(set(route_of))}
print(f"\nframes {N}; occluded-hazard frames {int(occ.sum())}", flush=True)

def ev(lg): return (F.softmax(lg, 1) * speeds_v.cpu()).sum(1)
def acc(lg, sel=None): c = (lg.argmax(1) == lab).float().numpy(); return float(100 * (c[sel].mean() if sel is not None else c.mean()))
def ce(lg, sel=None): c = F.cross_entropy(lg, lab, reduction="none").numpy(); return float(c[sel].mean() if sel is not None else c.mean())

# ---------------- D2 ----------------
print("\nD2 counterfactual probe (model, own tokens vs condition): argmax change %, mean E[v]_tok - E[v]_c (m/s), |dE|>1 %, tok slower among |dE|>1 %")
res["D2"] = {}
for c in ("none", "shuf", "rnd"):
    row = {}
    for part, sel in (("all", np.ones(N, bool)), ("occ", occ)):
        ch = (L["tok"].argmax(1) != L[c].argmax(1)).numpy()[sel]; dE = (ev(L["tok"]) - ev(L[c])).numpy()[sel]; big = np.abs(dE) > 1
        row[part] = dict(argmax_change=float(100 * ch.mean()), mean_dE=float(dE.mean()), big=float(100 * big.mean()),
                         slower_among_big=float(100 * (dE[big] < 0).mean()) if big.any() else float("nan"), n=int(sel.sum()))
        print(f"  tok vs {c:5s} {part:3s} n={row[part]['n']:5d}  change {row[part]['argmax_change']:5.1f}%  mean dE {row[part]['mean_dE']:+.3f}  "
              f"|dE|>1 {row[part]['big']:5.1f}%  slower {row[part]['slower_among_big']:5.1f}%")
    res["D2"][c] = row

# ---------------- accuracy table (D5 + reference) ----------------
print("\ntarget-speed accuracy % / CE: all | occluded")
res["acc"] = {}
for c in B_CONDS + M_CONDS:
    res["acc"][c] = dict(acc_all=acc(L[c]), acc_occ=acc(L[c], occ), ce_all=ce(L[c]), ce_occ=ce(L[c], occ))
    print(f"  {c:9s} {res['acc'][c]['acc_all']:5.1f} {res['acc'][c]['ce_all']:.3f} | {res['acc'][c]['acc_occ']:5.1f} {res['acc'][c]['ce_occ']:.3f}")
print("D5 content gap on occluded frames (acc tok - rnd): clean {:+.1f}  speed-swapped {:+.1f}  lidar-swapped {:+.1f} pt".format(
    *[res["acc"][t]["acc_occ"] - res["acc"][r]["acc_occ"] for t, r in (("tok", "rnd"), ("tok_spd", "rnd_spd"), ("tok_lid", "rnd_lid"))]))
res["D5"] = {k: res["acc"][t]["acc_occ"] - res["acc"][r]["acc_occ"] for k, t, r in (("gap_clean", "tok", "rnd"), ("gap_spd", "tok_spd", "rnd_spd"), ("gap_lid", "tok_lid", "rnd_lid"))}
res["D5"].update({f"dCE_{w}_{who}": res["acc"][f"{p}_{w}"]["ce_all"] - res["acc"][p]["ce_all"] for w in ("spd", "lid") for who, p in (("base", "base"), ("model", "tok"))})
print("D5 CE increase (all frames) when swapped: base speed {dCE_spd_base:+.3f} lidar {dCE_lid_base:+.3f}; model speed {dCE_spd_model:+.3f} lidar {dCE_lid_model:+.3f}".format(**res["D5"]))

# ---------------- D4 / D8 probes ----------------
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, r2_score
gkf = GroupKFold(n_splits=5)
def cv_pred(X, y, make, proba=False, sel=None):
    X, y, gr = (X, y, groups) if sel is None else (X[sel], y[sel], groups[sel])
    out = np.zeros((len(y), len(np.unique(y)))) if proba else np.zeros(len(y))
    for tr, te in gkf.split(X, y, gr):
        m = make().fit(X[tr], y[tr])
        if proba:
            p = m.predict_proba(X[te]); full = np.full((len(te), out.shape[1]), 1e-6); cls = np.searchsorted(np.unique(y), m.classes_); full[:, cls] = p
            out[te] = full / full.sum(1, keepdims=True)
        else: out[te] = m.predict(X[te])
    return out, y
logreg = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.02, max_iter=3000))
ridge = lambda: make_pipeline(StandardScaler(), Ridge(alpha=10.0))
y = lab.numpy(); tokX = np.concatenate([R["states"].numpy(), R["mask"].numpy()], 1)
print("\nD4 base residual: base on occluded vs other frames")
res["D4"] = dict(base_acc_occ=acc(L["base"], occ), base_acc_other=acc(L["base"], ~occ), base_ce_occ=ce(L["base"], occ), base_ce_other=ce(L["base"], ~occ))
print("  base acc occ {base_acc_occ:.1f} other {base_acc_other:.1f}; CE occ {base_ce_occ:.3f} other {base_ce_other:.3f}".format(**res["D4"]))
for name, X in (("A_base_feat", FT["b"]), ("B_base_feat+tokens", np.concatenate([FT["b"], tokX], 1))):
    P, yy = cv_pred(X, y, logreg, proba=True); cls = np.unique(y); yi = np.searchsorted(cls, yy)
    pr_acc = 100 * (P.argmax(1) == yi); pr_ce = -np.log(P[np.arange(len(yi)), yi])
    res["D4"][name] = dict(acc_all=float(pr_acc.mean()), acc_occ=float(pr_acc[occ].mean()), ce_all=float(pr_ce.mean()), ce_occ=float(pr_ce[occ].mean()))
    print(f"  probe {name:20s} held-out acc {pr_acc.mean():5.1f} occ {pr_acc[occ].mean():5.1f}; CE {pr_ce.mean():.3f} occ {pr_ce[occ].mean():.3f}")

print("\nD8 linear probes from the target-speed query feature (route-grouped 5-fold): ego speed R2, hidden-hazard AUC, hazard xy R2")
res["D8"] = {}
spd = R["speed"].numpy(); xyh = R["haz_xy"].numpy(); hsel = occ & np.isfinite(xyh).all(1)
for name in ("b", "m_tok", "m_none"):
    X = FT[name]; r = {}
    p, yy = cv_pred(X, spd, ridge); r["speed_R2"] = float(r2_score(yy, p))
    P, yy = cv_pred(X, occ.astype(int), logreg, proba=True); r["hazard_AUC"] = float(roc_auc_score(yy, P[:, 1]))
    xs = [cv_pred(X, xyh[:, j], ridge, sel=hsel) for j in (0, 1)]
    r["haz_xy_R2"] = float(np.mean([r2_score(yy, p) for p, yy in xs])) if hsel.sum() > 50 else float("nan")
    res["D8"][name] = r
    print(f"  {name:7s} speed R2 {r['speed_R2']:.3f}  hazard AUC {r['hazard_AUC']:.3f}  hazard xy R2 {r['haz_xy_R2']:.3f} (n {int(hsel.sum())})")

# ---------------- D6 ----------------
if d6["bev_sample"] and d6["coop_sample"]:
    Bm = torch.cat(d6["bev_sample"]).numpy(); Cm = torch.cat(d6["coop_sample"]).numpy(); mu = Bm.mean(0)
    U, S, Vt = np.linalg.svd(Bm - mu, full_matrices=False); ev_ = S ** 2 / (S ** 2).sum(); k = int(np.searchsorted(np.cumsum(ev_), 0.9) + 1)
    Cc = Cm - Cm.mean(0); proj = Cc @ Vt[:k].T; share = float((proj ** 2).sum() / (Cc ** 2).sum())
    Cb = Cm - mu; share_raw = float(((Cb @ Vt[:k].T) ** 2).sum() / (Cb ** 2).sum())
    res["D6"] = dict(bev_norm=float(np.mean(d6["bev_norm"])), spd_norm=float(np.mean(d6["spd_norm"])), coop_norm=float(np.mean(d6["coop_norm"])),
                     null_norm=float(np.mean(d6["null_norm"])) if d6["null_norm"] else float("nan"), k90=k, dim=int(Bm.shape[1]),
                     chance=k / Bm.shape[1], coop_energy_in_bev_pcs=share, coop_energy_in_bev_pcs_bev_centred=share_raw)
    print("\nD6 norms: BEV {bev_norm:.2f} speed {spd_norm:.2f} coop {coop_norm:.2f} null {null_norm:.2f}; BEV 90% PCs k={k90}/{dim} (chance {chance:.2f}); "
          "coop energy in BEV PCs {coop_energy_in_bev_pcs:.2f} (around the BEV mean {coop_energy_in_bev_pcs_bev_centred:.2f})".format(**res["D6"]))
if a.out:
    json.dump(res, open(a.out, "w"), indent=1)
print("POSTHOC_DIAG_DONE", flush=True)
