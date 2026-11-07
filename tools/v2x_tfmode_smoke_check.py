"""Checks after train_scripts/garage_tfmode_smoke.sh (training-mode study, 2026-11-06): one 1-epoch run per configuration.
  scratch : loss normaliser equals tfpp_v2x_000's (fork 776a079) for a plain use_v2x run
  post    : base checkpoint loads with only coop_* missing; every base tensor moved (all parameters train)
  cont    : base checkpoint loads with nothing missing (use_v2x 0)
  lr10    : two optimizer groups, token encoder lr = 10 x the rest
  frz     : every non-coop tensor (parameters AND buffers: BN statistics) bitwise equal to the base; coop_* moved
  shuf    : as frz; plus a unit test that training-mode tokens are the batch-rolled eval-mode tokens
  reinit / reinitc : post / cont + the planning decoder re-initialised (unit test: reset tensors = fresh init, rest = base)
  mdrop / mdropc   : post / cont + training-only BEV block hidden from the planning decoder (unit test of the mask)
"""
import argparse, glob, json, os, sys
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "team_code"))

ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--base", required=True)
ap.add_argument("--ref", required=True); ap.add_argument("--cfgs", default="scratch post cont lr10 frz shuf")
a = ap.parse_args(); ok = True
def check(cond, msg):
    global ok; print(("PASS " if cond else "FAIL ") + msg, flush=True); ok = ok and bool(cond)

base = torch.load(a.base, map_location="cpu")
ref_w = json.load(open(a.ref))["detailed_loss_weights"]
def nz(w): return {k: round(float(v), 6) for k, v in w.items() if float(v) != 0.0}
def model_of(c):
    f = sorted(glob.glob(os.path.join(a.out, c, "model_*.pth"))); return torch.load(f[-1], map_location="cpu") if f else None

cfgs = a.cfgs.split()
for c in cfgs:
    sd = model_of(c)
    if sd is None:
        check(False, f"{c}: no model file in {a.out}/{c}"); continue
    w = json.load(open(os.path.join(a.out, c, "config.json")))["detailed_loss_weights"]
    check(nz(w) == nz(ref_w), f"{c}: normalised loss weights = tfpp_v2x_000's {nz(ref_w)} (got {nz(w)})")
    coop = [k for k in sd if k.startswith("coop_")]
    other = [k for k in sd if not k.startswith("coop_")]
    if c not in ("cont", "reinitc", "mdropc"):
        check(len(coop) == 6, f"{c}: token encoder present, 6 tensors ({coop})")
    if c in ("post", "cont", "lr10", "frz", "shuf", "reinit", "reinitc", "mdrop", "mdropc"):
        missing = [k for k in sd if k not in base]
        check(sorted(missing) == sorted(coop), f"{c}: keys absent from the base are exactly coop_* ({missing[:5]})")
        unexpected = [k for k in base if k not in sd]
        check(not unexpected, f"{c}: no base key dropped ({unexpected[:5]})")
    if c in ("post", "lr10", "reinit", "reinitc", "mdrop", "mdropc"):
        fl = [k for k in other if sd[k].is_floating_point() and sd[k].numel() > 1]
        moved = sum(not torch.equal(sd[k], base[k]) for k in fl)
        check(moved > 0.9 * len(fl), f"{c}: base tensors trained ({moved}/{len(fl)} float tensors moved)")
    if c in ("frz", "shuf"):
        same = [k for k in other if torch.equal(sd[k], base[k])]
        check(len(same) == len(other), f"{c}: all {len(other)} non-coop tensors (params + BN buffers) bitwise equal to the base ({len(same)})")
    if c == "lr10":
        opt = torch.load(os.path.join(a.out, c, "optimizer_0000.pth"), map_location="cpu")
        g = opt["param_groups"]; lrs = [x["lr"] for x in g]; il = [x.get("initial_lr") for x in g]
        check(len(g) == 2 and abs(il[1] / il[0] - 10.0) < 1e-9 and abs(lrs[1] / max(lrs[0], 1e-30) - 10.0) < 1e-6,
              f"lr10: 2 groups, initial_lr {il}, lr after the epoch {lrs}")
        check(len(g[1]["params"]) == 6, f"lr10: token group holds the 6 coop tensors ({len(g[1]['params'])}: proj 4 + slot 1 + null 1)")
if "frz" in cfgs and "shuf" in cfgs:
    f, s = model_of("frz"), model_of("shuf")
    if f is not None and s is not None:
        check(any(not torch.equal(f[k], s[k]) for k in f if k.startswith("coop_")), "frz vs shuf: token encoders differ (different tokens seen)")

# unit test of the shuffle path on the model itself (no data needed)
import config as cfgmod, model as modmod
for sh in (0, 1):
    cfg = cfgmod.GlobalConfig(); cfg.initialize(root_dir=[], setting="all", use_v2x=1, v2x_shuffle_train=sh)
    net = modmod.LidarCenterNet(cfg)
    torch.manual_seed(0); bs, k = 4, cfg.v2x_k
    st = torch.randn(bs, k, cfg.v2x_state_dim); mk = (torch.rand(bs, k) > 0.3).float()
    net.train(); t_train = net.coop_tokens(st, mk, bs)
    net.eval(); t_eval = net.coop_tokens(st, mk, bs); t_roll = net.coop_tokens(torch.roll(st, 1, 0), torch.roll(mk, 1, 0), bs)
    if sh:
        check(torch.equal(t_train, t_roll) and not torch.equal(t_train, t_eval), "shuffle on: training tokens = another sample's, eval tokens untouched")
    else:
        check(torch.equal(t_train, t_eval), "shuffle off: training tokens = own tokens")

if any(c in cfgs for c in ("reinit", "reinitc")):   # reset_planner on the real base checkpoint
    for uv in (1, 0):
        cfg = cfgmod.GlobalConfig(); cfg.initialize(root_dir=[], setting="all", use_v2x=uv)
        torch.manual_seed(0); fresh = modmod.LidarCenterNet(cfg).state_dict()
        torch.manual_seed(1); net = modmod.LidarCenterNet(cfg); net.load_state_dict(base, strict=False)
        names = modmod.reset_planner(net, fresh); sd = net.state_dict()
        per = {p: sum(k.startswith(p) for k in names) for p in modmod.PLANNER_PREFIXES}
        check(all(per.values()), f"reinit use_v2x={uv}: every planner prefix matched ({per})")
        check(all(torch.equal(sd[k], fresh[k]) for k in names), f"reinit use_v2x={uv}: {len(names)} reset tensors = fresh init")
        rest = [k for k in sd if k in base and k not in names]
        check(all(torch.equal(sd[k], base[k].to(sd[k].dtype)) for k in rest), f"reinit use_v2x={uv}: other {len(rest)} base tensors untouched")
        check(not any(k.startswith(("backbone.", "coop_")) for k in names), f"reinit use_v2x={uv}: backbone / coop not reset")
if any(c in cfgs for c in ("mdrop", "mdropc")):   # the mask itself
    torch.manual_seed(0); h, w, n = 8, 8, 8 * 8 + 1 + 16
    m = modmod.mem_drop_mask(4000, (h, w), n, 1.0, 0.25, 0.75, "cpu"); fr = m[:, :h * w].float().mean(1)
    check(bool(m[:, h * w:].sum() == 0), "mdrop mask: speed/command and coop tokens never hidden")
    check(bool((fr > 0).all()) and float(fr.min()) >= 0.15 and float(fr.max()) <= 0.85, f"mdrop mask p=1: every sample hides a block, area {float(fr.min()):.2f}-{float(fr.max()):.2f} mean {float(fr.mean()):.2f}")
    g = m[:, :h * w].view(-1, h, w); rows, cols = g.any(2).sum(1), g.any(1).sum(1)
    check(bool((g.sum((1, 2)) == rows * cols).all()), "mdrop mask: hidden area is one rectangle")
    m5 = modmod.mem_drop_mask(4000, (h, w), n, 0.5, 0.25, 0.75, "cpu"); f5 = float(m5.any(1).float().mean())
    check(0.45 < f5 < 0.55, f"mdrop mask p=0.5: fraction of samples with a block {f5:.3f}")
    check(not modmod.mem_drop_mask(64, (h, w), n, 0.0, 0.25, 0.75, "cpu").any(), "mdrop mask p=0: nothing hidden")
    # inside the real forward: the checkpoint decoder gets the mask in train mode only; eval output = p 0 model's
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = cfgmod.GlobalConfig(); cfg.initialize(root_dir=[], setting="all", use_v2x=1, tfm_mem_drop_p=1.0)
    torch.manual_seed(0); net = modmod.LidarCenterNet(cfg).to(dev); net.load_state_dict(base, strict=False)
    cfg0 = cfgmod.GlobalConfig(); cfg0.initialize(root_dir=[], setting="all", use_v2x=1, tfm_mem_drop_p=0.0)
    net0 = modmod.LidarCenterNet(cfg0).to(dev); net0.load_state_dict(net.state_dict())
    seen = []
    net.join.register_forward_pre_hook(lambda mod, args, kw: seen.append((args[1].shape[1], kw.get("memory_key_padding_mask"))), with_kwargs=True)
    bs = 2; torch.manual_seed(3)
    inp = dict(rgb=torch.randn(bs, 3, cfg.camera_height, cfg.camera_width, device=dev),
               lidar_bev=torch.randn(bs, cfg.lidar_seq_len * (2 if cfg.use_ground_plane else 1), cfg.lidar_resolution_height, cfg.lidar_resolution_width, device=dev),
               target_point=torch.randn(bs, 2, device=dev), ego_vel=torch.rand(bs, 1, device=dev) * 10,
               command=torch.nn.functional.one_hot(torch.randint(0, 6, (bs,)), 6).float().to(dev),
               coop_states=torch.randn(bs, 16, 7, device=dev), coop_mask=torch.ones(bs, 16, device=dev))
    net.train(); net(**inp)
    nm, mk = seen[-1]; hw = nm - 1 - 16
    check(mk is not None and tuple(mk.shape) == (bs, nm) and bool(mk.any(1).all()) and not bool(mk[:, hw:].any()),
          f"mdrop forward (train, p=1): checkpoint decoder memory {nm} tokens = {hw} BEV + 1 + 16 coop; mask hides BEV only, area {mk[:, :hw].float().mean(1).tolist()}")
    net0.load_state_dict(net.state_dict())   # the train-mode pass above updated BN running statistics
    net.eval(); net0.eval(); seen.clear()
    with torch.no_grad():
        o, o0 = net(**inp), net0(**inp)
    check(seen and seen[-1][1] is None, "mdrop forward (eval): no mask passed")
    check(all(torch.equal(a_, b_) for a_, b_ in zip(o[:3], o0[:3]) if a_ is not None), "mdrop forward (eval): outputs bitwise = p 0 model")
print("TFMODE_SMOKE_PASSED" if ok else "TFMODE_SMOKE_FAILED")
