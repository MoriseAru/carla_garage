"""Checks after train_scripts/garage_tfmode_smoke.sh (training-mode study, 2026-11-06): one 1-epoch run per configuration.
  scratch : loss normaliser equals tfpp_v2x_000's (fork 776a079) for a plain use_v2x run
  post    : base checkpoint loads with only coop_* missing; every base tensor moved (all parameters train)
  cont    : base checkpoint loads with nothing missing (use_v2x 0)
  lr10    : two optimizer groups, token encoder lr = 10 x the rest
  frz     : every non-coop tensor (parameters AND buffers: BN statistics) bitwise equal to the base; coop_* moved
  shuf    : as frz; plus a unit test that training-mode tokens are the batch-rolled eval-mode tokens
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
    if c != "cont":
        check(len(coop) == 6, f"{c}: token encoder present, 6 tensors ({coop})")
    if c in ("post", "cont", "lr10", "frz", "shuf"):
        missing = [k for k in sd if k not in base]
        check(sorted(missing) == sorted(coop), f"{c}: keys absent from the base are exactly coop_* ({missing[:5]})")
        unexpected = [k for k in base if k not in sd]
        check(not unexpected, f"{c}: no base key dropped ({unexpected[:5]})")
    if c in ("post", "lr10"):
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
print("TFMODE_SMOKE_PASSED" if ok else "TFMODE_SMOKE_FAILED")
