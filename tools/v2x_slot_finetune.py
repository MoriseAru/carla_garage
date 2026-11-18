"""Short fine-tune of tfpp_v2x_000 for LOCAL's set-token check (HANDOVER 10169): only coop_proj and coop_null are trained,
every other parameter (including coop_slot_embedding) is frozen and the whole network stays in eval mode (BN statistics and
dropout as trained, the same treatment as train.py --v2x_freeze_base).
  --mode set : coop_slot_embedding.weight set to 0 before training (the decoder then reads the tokens as an unordered set)
  --mode ctl : coop_slot_embedding.weight kept as trained (same fine-tune, structure unchanged; control)
Loss = the two planning terms of the TF++ loss that depend on the tokens, with the run's weights:
  0.1 * loss_target_speed (model.loss_speed, two-hot targets) + 0.1 * loss_checkpoint (L1 over predict_checkpoint_len points).
(Detection / semantic / depth heads read the BEV features before the tokens join, so their losses carry no gradient here.)
Data: the full TF++ training set (setting 'all', the run's augmentation), frames in a seeded random order (seed 0, identical for
both modes), batch --bs, AdamW(amsgrad, lr --lr, weight decay 0.01 = train.py), --steps steps.
Writes <out>/model_0030.pth (state_dict with exactly the keys of the source), config.json, args.txt (copied from the source,
args.txt with the fine-tune settings appended under "finetune"), train log lines.
"""
import argparse, gzip, json, math, os, shutil, sys, time
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "team_code"))
import jsonpickle, jsonpickle.ext.numpy as jsonpickle_numpy; jsonpickle_numpy.register_handlers()
import config as cfgmod, model as modmod, data as datamod  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["set", "ctl"], required=True)
ap.add_argument("--train_dec", type=int, default=0, help="1: also train join, checkpoint_decoder, target_speed_network, "
                "checkpoint_query (LOCAL 10328); these modules are put in train mode")
ap.add_argument("--attn_w", type=float, default=0.0, help="> 0: decisive-vehicle attention loss (LOCAL 10589): speed query cross-attention, "
                "heads and layers averaged, renormalised over valid token slots; -log p[slot of speed_reduced_by_obj_id]")
ap.add_argument("--lon", type=int, default=0, help="1: longitudinal-only token routing masks in the decoder (tools/v2x_lon_mask.py)")
ap.add_argument("--src", default="/work/gn21/n21001/carla_garage_runs/tfpp_v2x_000")
ap.add_argument("--out", required=True, help="output dir; may contain {s}, replaced by the step count (needed with --save_at)")
ap.add_argument("--save_at", default="", help="comma-separated steps at which to also save a checkpoint (dose curve, LOCAL 10766)")
ap.add_argument("--root", default="/work/gn21/n21001/carla_garage_data_root")
ap.add_argument("--steps", type=int, default=2000); ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--sched", choices=["const", "cos"], default="const", help="cos: cosine decay from --lr to 0 over --steps, no warmup (LOCAL 11043)")
ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--workers", type=int, default=32)
ap.add_argument("--max_min", type=float, default=21.0, help="wall-clock guard for the training loop (minutes)")
a = ap.parse_args()
t0 = time.time(); torch.manual_seed(0); np.random.seed(0)
dev = "cuda"

cfg = cfgmod.GlobalConfig(); saved = jsonpickle.decode(open(os.path.join(a.src, "config.json")).read()); cfg.__dict__.update(saved.__dict__)
net = modmod.LidarCenterNet(cfg)
sd = torch.load(os.path.join(a.src, "model_0030.pth"), map_location="cpu")
res = net.load_state_dict(sd, strict=True); print("loaded", a.src, "strict ok", flush=True)
if a.mode == "set":
    with torch.no_grad():
        net.coop_slot_embedding.weight.zero_()
for n, p in net.named_parameters():
    p.requires_grad_(n.startswith("coop_proj.") or n == "coop_null" or (a.train_dec and (
        n.startswith(("join.", "checkpoint_decoder.", "target_speed_network.")) or n == "checkpoint_query")))
train_names = [n for n, p in net.named_parameters() if p.requires_grad]
print("trainable:", train_names, sum(p.numel() for p in net.parameters() if p.requires_grad), flush=True)
net.to(dev).eval()
for m_ in net.modules():   # as train.py --v2x_freeze_base: RNNs stay in train mode (cudnn has no eval-mode backward;
    if isinstance(m_, torch.nn.RNNBase):   # single-layer GRU without dropout -> identical output)
        m_.training = True
if a.train_dec:
    for mod in (net.join, net.checkpoint_decoder, net.target_speed_network):
        mod.train()
if a.lon:
    import v2x_lon_mask; v2x_lon_mask.install(net); print("lon-only decoder masks installed", flush=True)
slot_before = net.coop_slot_embedding.weight.detach().clone()
P_, K_ = int(cfg.predict_checkpoint_len), int(cfg.v2x_k)
cap = {"on": False, "att": []}
if a.attn_w > 0:
    for layer in net.join.layers:
        mha = layer.multihead_attn; orig = mha.forward

        def fwd(*args, _o=orig, _m=mha, **kw):
            r = _o(*args, **kw)                       # the normal pass (attention dropout as in training)
            if cap["on"]:                             # weights for the loss: same inputs, no attention dropout
                p0 = _m.dropout; _m.dropout = 0.0
                w = _o(*args, **dict(kw, need_weights=True, average_attn_weights=True))[1]
                _m.dropout = p0
                cap["att"].append(w[:, P_, -K_:])      # (bs, K): speed query on the token slots
            return r
        mha.forward = fwd

    import v2x_features   # record the slot -> actor id order of coop_states_from_boxes (same sort, same augmentation)
    _orig_csb = v2x_features.coop_states_from_boxes
    _LAST = {}

    def _csb(boxes, k=16, y_augmentation=0.0, yaw_augmentation=0.0, radius=64.0, **kw):
        out = _orig_csb(boxes, k=k, y_augmentation=y_augmentation, yaw_augmentation=yaw_augmentation, radius=radius, **kw)
        aug = math.radians(yaw_augmentation); c, s_ = math.cos(aug), math.sin(aug); cands = []
        for b in boxes:
            if b.get("class") != "car":
                continue
            px, py = float(b["position"][0]), float(b["position"][1]) - float(y_augmentation)
            d = math.hypot(c * px + s_ * py, -s_ * px + c * py)
            if d <= radius:
                cands.append((d, b.get("id")))
        cands.sort(key=lambda t: t[0]); _LAST["ids"] = [t[1] for t in cands[:k]]
        return out
    v2x_features.coop_states_from_boxes = _csb

cfg.initialize(root_dir=[a.root], setting="all", use_v2x=1, use_v2x_bev=0)
ds = datamod.CARLA_Data(root=cfg.data_roots, config=cfg, estimate_class_distributions=False, estimate_sem_distribution=False,
                        shared_dict=None, rank=0)
g = torch.Generator().manual_seed(0)
idx = torch.randperm(len(ds), generator=g)[: a.steps * a.bs].tolist()
class _DecSlot(torch.utils.data.Dataset):
    """adds dec_slot: token slot of the expert's speed_reduced_by_obj_id (-1 if none / not a token)."""
    def __init__(self, base, ids): self.b, self.ids = base, ids
    def __len__(self): return len(self.ids)
    def __getitem__(self, j):
        i = self.ids[j]; d = self.b[i]
        mf = str(self.b.measurements[i][0], encoding="utf-8") + f"/{self.b.sample_start[i] + cfg.seq_len - 1:04}.json.gz"
        with gzip.open(mf, "rt", encoding="utf-8") as f:
            oid = json.load(f).get("speed_reduced_by_obj_id")
        ids = _LAST.get("ids", [])
        d["dec_slot"] = ids.index(oid) if (oid is not None and oid in ids) else -1
        return d


dl = torch.utils.data.DataLoader(_DecSlot(ds, idx) if a.attn_w > 0 else torch.utils.data.Subset(ds, idx), batch_size=a.bs, shuffle=False, num_workers=a.workers,
                                 drop_last=True, worker_init_fn=lambda w: np.random.seed(1000 + w))
print(f"dataset {len(ds)} frames; using {len(idx)}; setup {time.time()-t0:.0f}s", flush=True)
save_at = sorted({int(x) for x in a.save_at.replace(":", ",").split(",") if x.strip()})   # ":" also accepted (qsub -v splits on ",")
assert not save_at or "{s}" in a.out, "--save_at needs {s} in --out"


def save(step):
    out = a.out.replace("{s}", str(step))
    assert torch.equal(net.coop_slot_embedding.weight.detach(), slot_before), "slot embedding changed"
    os.makedirs(out, exist_ok=True)
    out_sd = {k: v.detach().cpu() for k, v in net.state_dict().items()}
    assert list(out_sd.keys()) == list(sd.keys()) and all(out_sd[k].shape == sd[k].shape for k in sd), "state_dict keys differ"
    changed = sorted({k for k in sd if not torch.equal(out_sd[k], sd[k])})
    print(f"[save step {step}] {len(changed)} tensors changed vs source: {changed}", flush=True)
    torch.save(out_sd, os.path.join(out, "model_0030.pth"))
    shutil.copy(os.path.join(a.src, "config.json"), os.path.join(out, "config.json"))
    args_src = json.load(open(os.path.join(a.src, "args.txt")))
    args_src["id"] = os.path.basename(out.rstrip("/"))
    args_src["finetune"] = dict(vars(a), source=os.path.join(a.src, "model_0030.pth"), steps_done=step, trainable=train_names,
                                loss="0.1*loss_target_speed + 0.1*loss_checkpoint" + (f" + {a.attn_w}*decisive-vehicle attention NLL" if a.attn_w > 0 else ""), net_mode="eval (frozen BN/dropout)" + ("; join/checkpoint_decoder/target_speed_network in train mode" if a.train_dec else ""),
                                inference_note="needs V2X_LON_ONLY=1 (decoder masks)" if a.lon else "plain decoder")
    json.dump(args_src, open(os.path.join(out, "args.txt"), "w"), indent=2)
    print(f"SAVED step {step} -> {out} ({time.time()-t0:.0f}s)", flush=True)


opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=a.lr, amsgrad=True)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: 0.5 * (1.0 + math.cos(math.pi * k / a.steps))) if a.sched == "cos" else None   # update k uses lr * f(k): f(0)=1, f(steps)=0
w_ts, w_ck = float(cfg.detailed_loss_weights["loss_target_speed"]), float(cfg.detailed_loss_weights["loss_checkpoint"])
t1 = time.time(); step = 0; run = []; run_at = []
for data in dl:
    rgb = data["rgb"].to(dev, dtype=torch.float32); lidar = data["lidar"].to(dev, dtype=torch.float32)
    tp = data["target_point"].to(dev, dtype=torch.float32); cmd = data["command"].to(dev, dtype=torch.float32)
    vel = data["speed"].to(dev, dtype=torch.float32).unsqueeze(1)
    tw = data["target_speed_twohot"].to(dev, dtype=torch.float32)
    ck = data["route"][:, : cfg.predict_checkpoint_len].to(dev, dtype=torch.float32)
    st = data["coop_states"].to(dev, dtype=torch.float32); mk = data["coop_mask"].to(dev, dtype=torch.float32)
    cap["on"] = a.attn_w > 0; cap["att"] = []
    o = net(rgb=rgb, lidar_bev=lidar, target_point=tp, ego_vel=vel, command=cmd, target_point_next=None, coop_states=st, coop_mask=mk)
    l_ts = net.loss_speed(o[1], tw); l_ck = torch.mean(torch.abs(o[2] - ck))
    loss = w_ts * l_ts + w_ck * l_ck
    l_at = torch.zeros((), device=dev)
    if a.attn_w > 0:
        cap["on"] = False
        A = torch.stack(cap["att"], 0).mean(0)                     # (bs, K): heads and layers averaged
        valid = mk > 0.5; ds_ = data["dec_slot"].to(dev)
        sel = (ds_ >= 0) & valid.gather(1, ds_.clamp(min=0)[:, None]).squeeze(1)
        if bool(sel.any()):
            Am = A * valid; pdist = Am / Am.sum(1, keepdim=True).clamp(min=1e-12)
            l_at = -torch.log(pdist[sel].gather(1, ds_[sel][:, None]).squeeze(1).clamp(min=1e-8)).mean()
            loss = loss + a.attn_w * l_at
        run_at.append((float(l_at), float(sel.float().mean())))
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    if sched is not None:
        sched.step()
    step += 1; run.append((float(l_ts), float(l_ck)))
    if step % 100 == 0:
        m = np.mean(run[-100:], 0)
        ma = np.mean(run_at[-100:], 0) if run_at else (0.0, 0.0)
        print(f"step {step} ts {m[0]:.4f} ckpt {m[1]:.4f} attn {ma[0]:.4f} (frames {ma[1]:.2f}) lr {opt.param_groups[0]['lr']:.3e} {(time.time()-t1)/step:.3f}s/step", flush=True)
    if step in save_at and step != a.steps:
        save(step)
    if (time.time() - t1) / 60.0 > a.max_min:
        print(f"WALL GUARD hit at step {step}", flush=True); break
save(step)
print(f"FINETUNE_DONE {a.mode} steps {step} out {a.out} total {time.time()-t0:.0f}s", flush=True)
