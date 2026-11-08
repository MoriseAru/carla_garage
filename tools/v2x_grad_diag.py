"""D3 (HANDOVER 6717): gradient starvation of a post-hoc token path. No parameter is updated: forward + backward of the
planning losses (target-speed CE + checkpoint L1) on a few batches, eval-mode BN/dropout, for four starting points of the
same co-train plug-in model (use_v2x=1, tokens in the decoder memory, as 6380 / 6676):
  a_post   : base checkpoint + freshly initialised coop_* (the P tier at step 0)
  b_snp    : as a, but every base parameter shrunk and perturbed: w = lam * w_base + gam * w_fresh (Ash & Adams 2020)
  c_reinit : as a, but the planning decoder re-initialised (tfm_reinit_planner, the 6676 reinit run at step 0)
  d_scratch: everything freshly initialised (the S tier at step 0)
Reported per start: planning loss; gradient norm per parameter group (coop_*, join, rest of the planner, backbone);
per-token gradient norm on the decoder memory: BEV tokens vs the speed token vs valid coop tokens.
  python tools/v2x_grad_diag.py --root <data_root> --base <run_dir> [--batches 8] [--lam 0.4 --gam 0.1]
"""
import argparse, os, sys, json
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "team_code"))
import jsonpickle, jsonpickle.ext.numpy as jsonpickle_numpy; jsonpickle_numpy.register_handlers()
import config as cfgmod, model as modmod, data as datamod

ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--base", required=True)
ap.add_argument("--batches", type=int, default=8); ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--lam", type=float, default=0.4); ap.add_argument("--gam", type=float, default=0.1); ap.add_argument("--out", default="")
a = ap.parse_args(); dev = "cuda"

saved = jsonpickle.decode(open(os.path.join(a.base, "config.json")).read())
def make_cfg():
    cfg = cfgmod.GlobalConfig(); cfg.__dict__.update(saved.__dict__)
    cfg.initialize(root_dir=[a.root], setting="all", use_v2x=1); return cfg
cfg = make_cfg(); cfg.augment_percentage = 0.0; cfg.use_color_aug = 0; cfg.lidar_aug_prob = 0.0
ck = sorted(f for f in os.listdir(a.base) if f.startswith("model_") and f.endswith(".pth"))[-1]
base_sd = torch.load(os.path.join(a.base, ck), map_location="cpu"); print(f"base {a.base}/{ck}", flush=True)

ds = datamod.CARLA_Data(root=cfg.data_roots, config=cfg, estimate_class_distributions=False, estimate_sem_distribution=False, shared_dict=None, rank=0)
g = torch.Generator().manual_seed(0); idx = torch.randperm(len(ds), generator=g)[: a.batches * a.bs].tolist()
batches = list(torch.utils.data.DataLoader(torch.utils.data.Subset(ds, idx), batch_size=a.bs, shuffle=False, num_workers=8))


def build(start):
    torch.manual_seed(0); net = modmod.LidarCenterNet(cfg); fresh = {k: v.clone() for k, v in net.state_dict().items()}
    if start != "d_scratch":
        missing, unexpected = net.load_state_dict(base_sd, strict=False)
        assert all(k.startswith("coop_") for k in missing) and not unexpected, (missing[:5], unexpected[:5])
    if start == "b_snp":
        sd = net.state_dict(); params = dict(net.named_parameters())
        net.load_state_dict({k: (a.lam * sd[k] + a.gam * fresh[k]) if (k in params and not k.startswith("coop_")) else sd[k] for k in sd})
    if start == "c_reinit":
        modmod.reset_planner(net, fresh)
    return net.to(dev).eval()


def group(name):
    if name.startswith("coop_"): return "coop"
    if name.startswith("join."): return "join"
    if name.startswith(modmod.PLANNER_PREFIXES): return "planner_rest"
    if name.startswith("backbone."): return "backbone"
    return "other_heads"


res = {"lam": a.lam, "gam": a.gam, "frames": len(idx)}
for start in ("a_post", "b_snp", "c_reinit", "d_scratch"):
    net = build(start); mem_box = {}
    def pre(mod, args):
        args[1].retain_grad(); mem_box["mem"] = args[1]
    h = net.join.layers[0].register_forward_pre_hook(pre)
    sq = {}; tokg = {"bev": [], "spd": [], "coop": []}; losses = []
    for data in batches:
        net.zero_grad(set_to_none=True)
        rgb = data["rgb"].to(dev, dtype=torch.float32); lidar = data["lidar"].to(dev, dtype=torch.float32)
        tp = data["target_point"].to(dev, dtype=torch.float32); tpn = data["target_point_next"].to(dev, dtype=torch.float32) if cfg.two_tp_input else None
        vel = data["speed"].to(dev, dtype=torch.float32).unsqueeze(1); cmd = data["command"].to(dev, dtype=torch.float32)
        st = data["coop_states"].to(dev, dtype=torch.float32); mk = data["coop_mask"].to(dev, dtype=torch.float32)
        ts2 = data["target_speed_twohot"].to(dev, dtype=torch.float32); ckl = data["route"][:, : cfg.predict_checkpoint_len].to(dev, dtype=torch.float32)
        with torch.backends.cudnn.flags(enabled=False):   # cudnn GRU backward needs train mode; keep eval BN/dropout
          out = net(rgb=rgb, lidar_bev=lidar, target_point=tp, ego_vel=vel, command=cmd, target_point_next=tpn, coop_states=st, coop_mask=mk)
          loss = F.cross_entropy(out[1], ts2) + F.l1_loss(out[2], ckl)
          loss.backward()
        losses.append(float(loss))
        for n_, p in net.named_parameters():
            if p.grad is not None: sq[group(n_)] = sq.get(group(n_), 0.0) + float(p.grad.float().pow(2).sum())
        gm = mem_box["mem"].grad.float().norm(dim=-1)   # (bs, 81)
        tokg["bev"] += gm[:, :64].mean(1).tolist(); tokg["spd"] += gm[:, 64].tolist()
        v = mk > 0.5
        if v.any(): tokg["coop"] += gm[:, -cfg.v2x_k:][v].tolist()
    h.remove()
    r = {"loss": float(np.mean(losses)), "grad_norm": {k: float(np.sqrt(v / len(batches))) for k, v in sq.items()},
         "mem_token_grad": {k: float(np.mean(v)) for k, v in tokg.items() if v}}
    r["coop_over_bev_token_grad"] = r["mem_token_grad"]["coop"] / r["mem_token_grad"]["bev"]
    res[start] = r
    print(f"{start:9s} loss {r['loss']:.3f}  grad norm " + "  ".join(f"{k} {v:.2e}" for k, v in sorted(r["grad_norm"].items())) +
          "  | memory-token grad: " + "  ".join(f"{k} {v:.2e}" for k, v in r["mem_token_grad"].items()) + f"  coop/bev {r['coop_over_bev_token_grad']:.2f}", flush=True)
    del net; torch.cuda.empty_cache()
if a.out: json.dump(res, open(a.out, "w"), indent=1)
print("GRAD_DIAG_DONE", flush=True)
