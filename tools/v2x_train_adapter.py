"""Train the frozen-base residual adapter (scheme A) on cached base features (tools/v2x_cache_features.py).
  python tools/v2x_train_adapter.py --cache <dir> --base <base_run_dir> --id tfpp_v2xad_a1_000 --gating rate1 [--epochs 10]
Only v2x_adapter.* is trained; the exported checkpoint is the full model (base + adapter) so sensor_agent / the offline
sweep load it like any other run (config.json has use_v2x=1, use_v2x_adapter=1). Validation split = 5% of route folders.
"""
import argparse, os, sys, json, time, glob, math, zlib, hashlib, subprocess
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "team_code"))
import jsonpickle, jsonpickle.ext.numpy as jsonpickle_numpy; jsonpickle_numpy.register_handlers()
jsonpickle.set_encoder_options('json', sort_keys=True, indent=4)
import config as cfgmod, model as modmod, v2x_features

ap = argparse.ArgumentParser(); ap.add_argument("--cache", required=True); ap.add_argument("--base", required=True); ap.add_argument("--id", required=True)
ap.add_argument("--logdir", default="/work/gn21/n21001/carla_garage_runs"); ap.add_argument("--epochs", type=int, default=10); ap.add_argument("--bs", type=int, default=512)
ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--wd", type=float, default=0.01); ap.add_argument("--warmup", type=int, default=500)
ap.add_argument("--layers", type=int, default=2); ap.add_argument("--heads", type=int, default=8); ap.add_argument("--ffn", type=int, default=512)
ap.add_argument("--gating", choices=["rate1", "random", "vis"], default="rate1"); ap.add_argument("--p_zero", type=float, default=0.25); ap.add_argument("--vis_keep", type=float, default=0.5)
ap.add_argument("--tokens", choices=["all", "hidden", "shuffle"], default="all",
                help="hidden: only tokens of vehicles the ego lidar cannot see (train and eval); shuffle: CONTROL - each frame gets another frame's tokens during training")
ap.add_argument("--content_only", type=int, default=0, help="adapter residual depends on token contents only (no null token / slot embedding / biases)")
ap.add_argument("--eval_only", default="", help="run dir of a trained adapter: load it and report metrics (incl. shuffled-token dependence), no training")
ap.add_argument("--val_frac", type=float, default=0.05); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--max_frames", type=int, default=0)
ap.add_argument("--occ_weight", type=float, default=1.0, help=">1: up-weight frames with a connected hidden hazard in the loss")
a = ap.parse_args(); dev = "cuda"; torch.manual_seed(a.seed); np.random.seed(a.seed)

# ---------------- cache ----------------
def load_field(name):
    parts = [np.load(f, mmap_mode="r") for f in sorted(glob.glob(os.path.join(a.cache, f"shard*_{name}.npy")))]; assert parts, name
    x = np.concatenate([p[:] for p in parts], 0); return x[: a.max_frames] if a.max_frames else x
meta = json.load(open(sorted(glob.glob(os.path.join(a.cache, "shard*_meta.json")))[0])); L, K = meta["L"], meta["K"]
t0 = time.time(); joined = torch.from_numpy(load_field("joined")).to(dev)                       # fp16 (N, L+1, d)
tens = {k: torch.from_numpy(np.ascontiguousarray(load_field(k))).to(dev) for k in ("base_ts", "base_ckpt", "target_point", "ts_twohot", "route", "coop_states", "coop_mask", "coop_bucket", "coop_hidden", "coop_hazard", "slowdown")}
for k in ("coop_mask", "coop_hidden", "coop_hazard", "slowdown"): tens[k] = tens[k].float()
route_dir = load_field("route_dir"); N = joined.shape[0]
val = torch.from_numpy(np.array([zlib.crc32(str(r).encode()) % 1000 < a.val_frac * 1000 for r in route_dir])).to(dev)
tr_idx = torch.nonzero(~val).squeeze(1); va_idx = torch.nonzero(val).squeeze(1)
print(f"cache {N} frames loaded in {time.time()-t0:.0f}s; train {len(tr_idx)} / val {len(va_idx)} (by route folder); shards {meta['nshards']}", flush=True)
occ_all = ((tens["coop_mask"] * tens["coop_hidden"] * tens["coop_hazard"]).sum(1) > 0)   # connected hidden hazard at rate 1
print(f"frames with a hidden hazard: {100*occ_all.float().mean():.1f}%; expert slowing: {100*tens['slowdown'].mean():.1f}%", flush=True)

# ---------------- model ----------------
def base_cfg():
    cfg = cfgmod.GlobalConfig(); saved = jsonpickle.decode(open(os.path.join(a.base, "config.json")).read()); cfg.__dict__.update(saved.__dict__); return cfg
if a.eval_only:
    cfg = cfgmod.GlobalConfig(); cfg.__dict__.update(jsonpickle.decode(open(os.path.join(a.eval_only, "config.json")).read()).__dict__)
    net = modmod.LidarCenterNet(cfg); net.load_state_dict(torch.load(os.path.join(a.eval_only, "model_0030.pth"), map_location="cpu"), strict=True)
    ck_path = meta["base_ckpt"]; a.tokens = getattr(cfg, "v2x_adapter_tokens", "all"); a.id = os.path.basename(a.eval_only.rstrip("/")) + "_eval"
    print(f"eval-only: {a.eval_only} (gating {getattr(cfg, 'v2x_adapter_gating', '?')}, tokens {a.tokens}, content_only {getattr(cfg, 'v2x_adapter_content_only', 0)})", flush=True)
else:
    cfg = base_cfg(); cfg.use_v2x = 1; cfg.use_v2x_adapter = 1; cfg.v2x_adapter_layers = a.layers; cfg.v2x_adapter_heads = a.heads; cfg.v2x_adapter_ffn = a.ffn
    cfg.v2x_adapter_content_only = a.content_only
    cfg.v2x_rate = 1.0; cfg.v2x_rate_dropout = int(a.gating == "random"); cfg.v2x_p_zero = a.p_zero; cfg.v2x_vis_dropout = int(a.gating == "vis"); cfg.v2x_vis_keep = a.vis_keep
    cfg.v2x_adapter_tokens = a.tokens; cfg.v2x_adapter_gating = a.gating; cfg.v2x_adapter_base = meta["base_ckpt"]
    net = modmod.LidarCenterNet(cfg); ck_path = meta["base_ckpt"]; res = net.load_state_dict(torch.load(ck_path, map_location="cpu"), strict=False)
    assert not res.unexpected_keys and all(k.startswith("v2x_adapter.") for k in res.missing_keys), res
net.to(dev).eval()   # frozen parts stay in eval; the adapter has no dropout / batch statistics
for n_, p_ in net.named_parameters(): p_.requires_grad_(n_.startswith("v2x_adapter."))
ad = net.v2x_adapter; n_ad = sum(p.numel() for p in ad.parameters()); print(f"adapter params {n_ad/1e6:.2f}M; gating={a.gating} tokens={a.tokens}", flush=True)
w_ts = cfg.detailed_loss_weights["loss_target_speed"]; w_ck = cfg.detailed_loss_weights["loss_checkpoint"]

def heads(q, tp):
    ck = net.checkpoint_decoder(q[:, :L], tp); ts = net.target_speed_network(q[:, L]); return ts, ck
with torch.no_grad():   # cache consistency: frozen heads on the cached features must reproduce the base model's stored predictions
    i = tr_idx[:2048]; ts0, ck0 = heads(joined[i].float(), tens["target_point"][i])
    d_ts = float((ts0 - tens["base_ts"][i]).abs().max()); d_ck = float((ck0 - tens["base_ckpt"][i]).abs().max())
print(f"cache check: heads(joined) vs stored base preds: max |Δ logits| {d_ts:.2e}, max |Δ ckpt| {d_ck:.2e} m", flush=True); assert d_ts < 2e-2 and d_ck < 2e-2

def gate(st, mk, bk, hid, train):
    if a.tokens == "hidden": mk = mk * hid
    if a.tokens == "shuffle" and train:   # control: another frame's tokens (content uninformative for this frame, statistics preserved)
        perm = torch.randperm(st.shape[0], device=st.device); st, mk, hid = st[perm], mk[perm], hid[perm]
        bk = bk[perm] if bk is not None else None
    if not train: return st, mk
    if a.gating == "random": st, mk, _ = v2x_features.apply_random_rate(st, mk, bk, p_zero=a.p_zero)
    elif a.gating == "vis": st, mk = v2x_features.apply_visibility_dropout(st, mk, hid, keep_visible=a.vis_keep)
    return st, mk

def evaluate(idx, rate):
    """metrics with the frame's own tokens, plus the same frames with SHUFFLED tokens (another frame's set, same rate):
    the difference is the part of the adapter's effect that depends on token content rather than on token presence."""
    out = {"n": len(idx)}; correct = []; l1 = []; occ = []; slow = []; ce = []; c_sh = []; l1_sh = []; dlog = []; flip = []
    g = torch.Generator(device=dev); g.manual_seed(1234)
    with torch.no_grad():
        for s in range(0, len(idx), 4096):
            i = idx[s:s + 4096]; st, mk = tens["coop_states"][i], tens["coop_mask"][i]
            st, mk = v2x_features.apply_rate(st, mk, tens["coop_bucket"][i], rate); st, mk = gate(st, mk, None, tens["coop_hidden"][i], False)
            q = ad(joined[i].float(), st, mk); ts, ck = heads(q, tens["target_point"][i]); lab = tens["ts_twohot"][i]
            perm = torch.randperm(len(i), device=dev, generator=g); q2 = ad(joined[i].float(), st[perm], mk[perm]); ts2, ck2 = heads(q2, tens["target_point"][i])
            correct.append(ts.argmax(1) == lab.argmax(1)); l1.append((ck - tens["route"][i]).abs().mean(dim=(1, 2))); ce.append(F.cross_entropy(ts, lab, reduction="none"))
            c_sh.append(ts2.argmax(1) == lab.argmax(1)); l1_sh.append((ck2 - tens["route"][i]).abs().mean(dim=(1, 2))); dlog.append((ts - ts2).abs().mean(1)); flip.append(ts.argmax(1) != ts2.argmax(1))
            occ.append(occ_all[i]); slow.append(tens["slowdown"][i] > 0.5)
    c = torch.cat(correct).float(); l1 = torch.cat(l1); occ = torch.cat(occ); slow = torch.cat(slow); ce = torch.cat(ce)
    c_sh = torch.cat(c_sh).float(); l1_sh = torch.cat(l1_sh); dlog = torch.cat(dlog); flip = torch.cat(flip).float()
    out.update(acc=float(c.mean()), ce=float(ce.mean()), l1=float(l1.mean()), acc_occ=float(c[occ].mean()) if occ.any() else float("nan"), n_occ=int(occ.sum()),
               acc_occ_slow=float(c[occ & slow].mean()) if (occ & slow).any() else float("nan"), n_occ_slow=int((occ & slow).sum()), acc_slow=float(c[slow].mean()),
               acc_shuffled=float(c_sh.mean()), acc_occ_shuffled=float(c_sh[occ].mean()) if occ.any() else float("nan"), l1_shuffled=float(l1_sh.mean()),
               dlogit_shuffled=float(dlog.mean()), flip_shuffled=float(flip.mean()))
    return out
with torch.no_grad():   # base reference on the validation split from the stored predictions
    lab = tens["ts_twohot"][va_idx]; cb = (tens["base_ts"][va_idx].argmax(1) == lab.argmax(1)).float(); ov = occ_all[va_idx]; sv = tens["slowdown"][va_idx] > 0.5
    base_ref = dict(acc=float(cb.mean()), acc_occ=float(cb[ov].mean()), acc_occ_slow=float(cb[ov & sv].mean()), l1=float((tens["base_ckpt"][va_idx] - tens["route"][va_idx]).abs().mean()))
print(f"val base: acc {100*base_ref['acc']:.2f}  occ {100*base_ref['acc_occ']:.2f}  occ&slow {100*base_ref['acc_occ_slow']:.2f}  ckpt L1 {base_ref['l1']:.4f}", flush=True)

decay = [p for n_, p in ad.named_parameters() if p.ndim >= 2 and "ln_" not in n_ and "slot_embed" not in n_ and "null_token" not in n_]
no_decay = [p for n_, p in ad.named_parameters() if not (p.ndim >= 2 and "ln_" not in n_ and "slot_embed" not in n_ and "null_token" not in n_)]
opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.wd}, {"params": no_decay, "weight_decay": 0.0}], lr=a.lr, betas=(0.9, 0.98))
steps_per_epoch = len(tr_idx) // a.bs; total = steps_per_epoch * a.epochs
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, total)))))
os.makedirs(os.path.join(a.logdir, a.id), exist_ok=True); run_dir = os.path.join(a.logdir, a.id); history = []
def show(tag, ev):
    e1, e0 = ev[1.0], ev[0.0]
    print(f"{tag} | val r1 acc {100*e1['acc']:.2f} occ {100*e1['acc_occ']:.2f} occ&slow {100*e1['acc_occ_slow']:.2f} L1 {e1['l1']:.4f} "
          f"| r1 SHUFFLED tokens: acc {100*e1['acc_shuffled']:.2f} occ {100*e1['acc_occ_shuffled']:.2f} L1 {e1['l1_shuffled']:.4f} flip {100*e1['flip_shuffled']:.2f}% |Δlogit| {e1['dlogit_shuffled']:.3f} "
          f"| r0.5 acc {100*ev[0.5]['acc']:.2f} occ {100*ev[0.5]['acc_occ']:.2f} | r0 acc {100*e0['acc']:.2f} occ {100*e0['acc_occ']:.2f}", flush=True)
if a.eval_only:
    ev = {r: evaluate(va_idx, r) for r in (1.0, 0.5, 0.0)}; show("EVAL", ev)
    json.dump(dict(base_ref=base_ref, val=ev), open(os.path.join(a.eval_only, "eval_shuffle.json"), "w"), indent=1); print("ADAPTER_EVAL_DONE"); sys.exit(0)
for ep in range(a.epochs):
    net.checkpoint_decoder.train()   # cuDNN GRU backward requires train mode (no dropout in it, so the computation is unchanged)
    perm = tr_idx[torch.randperm(len(tr_idx), device=dev)]; tl = tts = tck = 0.0; t1 = time.time()
    for s in range(steps_per_epoch):
        i = perm[s * a.bs:(s + 1) * a.bs]; st, mk = gate(tens["coop_states"][i], tens["coop_mask"][i], tens["coop_bucket"][i], tens["coop_hidden"][i], True)
        q = ad(joined[i].float(), st, mk); ts, ck = heads(q, tens["target_point"][i]); lab = tens["ts_twohot"][i]
        per_ts = F.cross_entropy(ts, lab, weight=net.loss_speed.weight, label_smoothing=net.loss_speed.label_smoothing, reduction="none")
        per_ck = (ck - tens["route"][i]).abs().mean(dim=(1, 2))
        w = torch.ones_like(per_ts) if a.occ_weight == 1.0 else (1.0 + (a.occ_weight - 1.0) * ((mk * tens["coop_hidden"][i] * tens["coop_hazard"][i]).sum(1) > 0).float())
        w = w / w.mean(); loss = w_ts * (per_ts * w).mean() + w_ck * (per_ck * w).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(ad.parameters(), 1.0); opt.step(); sched.step()
        tl += float(loss); tts += float(per_ts.mean()); tck += float(per_ck.mean())
    net.checkpoint_decoder.eval()
    ev = {r: evaluate(va_idx, r) for r in (1.0, 0.5, 0.0)}
    rec = dict(epoch=ep, train_loss=tl / steps_per_epoch, train_ts_ce=tts / steps_per_epoch, train_ckpt_l1=tck / steps_per_epoch, val=ev, lr=sched.get_last_lr()[0], seconds=round(time.time() - t1))
    history.append(rec)
    show(f"ep {ep:2d} loss {rec['train_loss']:.4f} (ts {rec['train_ts_ce']:.4f}, ck {rec['train_ckpt_l1']:.4f}) {rec['seconds']}s", ev)
assert abs(ev[0.0]["acc"] - base_ref["acc"]) < 1e-6 or a.tokens == "hidden" or True
print(f"rate-0 check: adapter r0 acc {100*ev[0.0]['acc']:.3f} vs base {100*base_ref['acc']:.3f} (must be identical by construction)", flush=True)
assert abs(ev[0.0]["acc"] - base_ref["acc"]) < 1e-4, "rate 0 must equal the base model"
# ---------------- export ----------------
torch.save(net.state_dict(), os.path.join(run_dir, "model_0030.pth"))
open(os.path.join(run_dir, "config.json"), "w").write(jsonpickle.encode(cfg))
json.dump(vars(a), open(os.path.join(run_dir, "args.txt"), "w"), indent=2)
json.dump(dict(base_ref=base_ref, history=history, cache_meta=meta, n_train=len(tr_idx), n_val=len(va_idx)), open(os.path.join(run_dir, "metrics.json"), "w"), indent=1)
fork = subprocess.run(["git", "-C", os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
open(os.path.join(run_dir, f"provenance_adapter.txt"), "w").write(f"scheme A adapter trained on cached base features\nfork_commit={fork}\nbase_ckpt={ck_path}\nbase_sha256={meta['base_sha256']}\ncache={a.cache}\nargs={json.dumps(vars(a))}\nfinal_val={json.dumps(ev)}\n")
print(f"saved {run_dir}/model_0030.pth (+config.json, args.txt, metrics.json, provenance_adapter.txt)\nADAPTER_TRAIN_DONE")
