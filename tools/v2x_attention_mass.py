"""Attention-mass diagnostic: how much of the planning queries' cross-attention lands on the cooperative tokens?
  python tools/v2x_attention_mass.py --root <data_root> --runs orig=<dir> rd=<dir> ... [--n 2000] [--rates "1.0 0.5"]
For memory-token models (original / rd / occ / dual*) the decoder layers' attention weights are captured; the mass on
the K coop positions is split into valid vs null slots, and per-valid-slot mass is compared between hidden-hazard
tokens and visible ones. For adapter models (use_v2x_adapter) the adapter's own attention (null vs tokens) and the
residual norm relative to the query norm are reported. Strata: frames with a connected hidden hazard (occ) vs rest.
"""
import argparse, os, sys, json
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "team_code"))
import jsonpickle, jsonpickle.ext.numpy as jsonpickle_numpy; jsonpickle_numpy.register_handlers()
import config as cfgmod, model as modmod, data as datamod, v2x_features

ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--runs", required=True, nargs="+", help="name=run_dir ... (space separated: PBS -v splits on commas)")
ap.add_argument("--n", type=int, default=2000); ap.add_argument("--bs", type=int, default=16); ap.add_argument("--rates", default="1.0 0.5")
ap.add_argument("--out", default="/work/gn21/n21001/V2XState_Real/tmp/v2x_attention_mass.json")
ap.add_argument("--tokens", default="all", choices=["all", "hidden", "visible"],
                help="offline counterpart of V2X_TOKENS: restrict the token set to the vehicles the ego lidar cannot / can see")
a = ap.parse_args(); dev = "cuda"
runs = dict(kv.split("=") for kv in a.runs); rates = [float(r) for r in a.rates.split()]

def load(run_dir):
    cfg = cfgmod.GlobalConfig(); saved = jsonpickle.decode(open(os.path.join(run_dir, "config.json")).read()); cfg.__dict__.update(saved.__dict__)
    net = modmod.LidarCenterNet(cfg); ck = sorted(f for f in os.listdir(run_dir) if f.startswith("model_") and f.endswith(".pth"))[-1]
    net.load_state_dict(torch.load(os.path.join(run_dir, ck), map_location="cpu"), strict=False); net.to(dev).eval(); return cfg, net

cfg0, _ = load(next(iter(runs.values())))
cfg0.initialize(root_dir=[a.root], setting="all", use_v2x=1)
ds = datamod.CARLA_Data(root=cfg0.data_roots, config=cfg0, estimate_class_distributions=False, estimate_sem_distribution=False, shared_dict=None, rank=0)
g = torch.Generator().manual_seed(0); idx = torch.randperm(len(ds), generator=g)[: a.n].tolist()
dl = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, idx), batch_size=a.bs, shuffle=False, num_workers=16)
print(f"frames {len(idx)}; runs {list(runs)}; rates {rates}; token subset {a.tokens}", flush=True)
L = cfg0.predict_checkpoint_len; K = cfg0.v2x_k
batches = []   # cache the batches so every run sees identical frames
for data in dl: batches.append({k: data[k] for k in ("rgb", "lidar", "target_point", "speed", "command", "coop_states", "coop_mask", "coop_bucket", "coop_hidden", "coop_hazard", "target_speed_twohot")})
print("batches cached", flush=True)

def hook_decoder(net):
    store = []
    for layer in net.join.layers:
        orig = layer.multihead_attn.forward
        def wrapped(*args, _orig=orig, **kw):
            kw["need_weights"] = True; kw["average_attn_weights"] = True; out = _orig(*args, **kw); store.append(out[1].detach()); return out
        layer.multihead_attn.forward = wrapped
    return store

results = {}
for name, run_dir in runs.items():
    cfg, net = load(run_dir); adapter = bool(getattr(net, "v2x_adapter_on", False))
    store = None if adapter else hook_decoder(net)
    acc = {}
    for rate in rates:
        agg = {s: {"n": 0, "ts_valid": 0.0, "ts_null": 0.0, "ck_valid": 0.0, "ck_null": 0.0, "per_hidden": 0.0, "n_hidden": 0, "per_visible": 0.0, "n_visible": 0,
                   "delta_rel": 0.0, "correct": 0} for s in ("occ", "rest")}
        for data in batches:
            rgb = data["rgb"].to(dev, dtype=torch.float32); lidar = data["lidar"].to(dev, dtype=torch.float32); tp = data["target_point"].to(dev, dtype=torch.float32)
            vel = data["speed"].to(dev, dtype=torch.float32).unsqueeze(1); cmd = data["command"].to(dev, dtype=torch.float32)
            st = data["coop_states"].to(dev, dtype=torch.float32); mk0 = data["coop_mask"].to(dev, dtype=torch.float32); bk = data["coop_bucket"].to(dev)
            hid = data["coop_hidden"].to(dev, dtype=torch.float32); haz = data["coop_hazard"].to(dev, dtype=torch.float32)
            st, mk = v2x_features.apply_rate(st, mk0, bk, rate); bs = rgb.shape[0]
            if a.tokens == "hidden": mk = mk * hid            # only vehicles with <= v2x_hidden_pts lidar points
            elif a.tokens == "visible": mk = mk * (1.0 - hid)
            occ = ((mk * hid * haz).sum(1) > 0)
            ts_label = data["target_speed_twohot"].to(dev).argmax(1)
            with torch.no_grad():
                if store is not None: store.clear()
                out = net(rgb=rgb, lidar_bev=lidar, target_point=tp, ego_vel=vel, command=cmd, coop_states=st, coop_mask=mk) if not adapter else None
                if adapter:
                    # run forward with adapter attention weights: hook the adapter call
                    net.v2x_adapter.forward.__func__  # noqa
                    q_store = {}
                    h = net.join.register_forward_hook(lambda m, inp, o: q_store.update(joined=o.detach()))
                    out = net(rgb=rgb, lidar_bev=lidar, target_point=tp, ego_vel=vel, command=cmd, coop_states=st, coop_mask=mk); h.remove()
                    _ = net.v2x_adapter(q_store["joined"], st, mk, need_weights=True); w_layers = net.v2x_adapter.last_attn
                    delta = net.v2x_adapter.last_delta; rel = (delta.norm(dim=-1) / q_store["joined"].norm(dim=-1).clamp(min=1e-6)).mean(1)   # (bs,)
                    W = torch.stack(w_layers, 0).mean(0)                     # (bs, Lq, 1+K): null first (K when content_only: no null token)
                    valid = (mk > 0.5).float()
                    if getattr(net.v2x_adapter, "content_only", False):
                        ts_valid = (W[:, L] * valid).sum(1); ts_null = torch.zeros_like(ts_valid)
                        ck_valid = (W[:, :L] * valid[:, None]).sum(-1).mean(1); ck_null = torch.zeros_like(ck_valid); per_slot = W[:, L]
                    else:
                        ts_valid = (W[:, L, 1:] * valid).sum(1); ts_null = W[:, L, 0]
                        ck_valid = (W[:, :L, 1:] * valid[:, None]).sum(-1).mean(1); ck_null = W[:, :L, 0].mean(1)
                        per_slot = W[:, L, 1:]
                else:
                    W = torch.stack(store, 0).mean(0)                        # (bs, Lq, S): coop tokens are the last K positions
                    valid = (mk > 0.5).float(); Wc = W[:, :, -K:]
                    ts_valid = (Wc[:, L] * valid).sum(1); ts_null = (Wc[:, L] * (1 - valid)).sum(1)
                    ck_valid = (Wc[:, :L] * valid[:, None]).sum(-1).mean(1); ck_null = (Wc[:, :L] * (1 - valid[:, None])).sum(-1).mean(1)
                    per_slot = Wc[:, L]; rel = torch.zeros(bs, device=dev)
                hidden_slot = valid * hid * haz; visible_slot = valid * (1 - hid)
                correct = (out[1].argmax(1) == ts_label)
            for s, sel in (("occ", occ), ("rest", ~occ)):
                if int(sel.sum()) == 0: continue
                A = agg[s]; A["n"] += int(sel.sum()); A["ts_valid"] += float(ts_valid[sel].sum()); A["ts_null"] += float(ts_null[sel].sum())
                A["ck_valid"] += float(ck_valid[sel].sum()); A["ck_null"] += float(ck_null[sel].sum()); A["delta_rel"] += float(rel[sel].sum()); A["correct"] += int(correct[sel].sum())
                A["per_hidden"] += float((per_slot * hidden_slot)[sel].sum()); A["n_hidden"] += int(hidden_slot[sel].sum())
                A["per_visible"] += float((per_slot * visible_slot)[sel].sum()); A["n_visible"] += int(visible_slot[sel].sum())
        acc[rate] = {s: dict(n=A["n"], ts_valid=A["ts_valid"] / max(A["n"], 1), ts_null=A["ts_null"] / max(A["n"], 1), ck_valid=A["ck_valid"] / max(A["n"], 1),
                             ck_null=A["ck_null"] / max(A["n"], 1), per_hidden=A["per_hidden"] / max(A["n_hidden"], 1), per_visible=A["per_visible"] / max(A["n_visible"], 1),
                             n_hidden=A["n_hidden"], n_visible=A["n_visible"], delta_rel=A["delta_rel"] / max(A["n"], 1), acc=A["correct"] / max(A["n"], 1)) for s, A in agg.items()}
    results[name] = {"adapter": adapter, "rates": {str(r): v for r, v in acc.items()}}
    kind = "adapter attn (null | tokens)" if adapter else "decoder attn on coop positions (valid | null)"
    print(f"\n== {name} [{kind}]  ({'S=' + str(W.shape[-1])})")
    print(f"{'rate':>5s} {'stratum':8s} {'n':>5s} {'ts→valid':>9s} {'ts→null':>8s} {'ck→valid':>9s} {'ck→null':>8s} {'per hidden tok':>14s} {'per visible tok':>15s} {'hid/vis':>8s} {'|Δ|/|q|':>8s} {'ts acc%':>8s}")
    for r, d in acc.items():
        for s, v in d.items():
            ratio = v["per_hidden"] / v["per_visible"] if v["per_visible"] > 0 else float("nan")
            print(f"{r:5.2f} {s:8s} {v['n']:5d} {v['ts_valid']:9.3f} {v['ts_null']:8.3f} {v['ck_valid']:9.3f} {v['ck_null']:8.3f} {v['per_hidden']:14.4f} {v['per_visible']:15.4f} {ratio:8.2f} {v['delta_rel']:8.3f} {100*v['acc']:8.1f}")
    del net; torch.cuda.empty_cache()
os.makedirs(os.path.dirname(a.out), exist_ok=True); json.dump(results, open(a.out, "w"), indent=1); print(f"\nwrote {a.out}\nATTENTION_MASS_DONE")
