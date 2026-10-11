"""Structural checks for scheme R ("reverse t3"): co-trained plug-in fine-tuned with KD to itself on token samples and
KD to the frozen base on zero-token samples. Exercises the REAL train.py code path (Engine.load_data_compute_loss).
  python tools/v2x_schemeR_smoke.py --root <smoke_root> --plugin <run_dir> --base <run_dir>
"""
import argparse, os, sys, types, tempfile, glob
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "team_code"))
import jsonpickle, jsonpickle.ext.numpy as jsonpickle_numpy; jsonpickle_numpy.register_handlers()
from config import GlobalConfig
from model import LidarCenterNet
import data as datamod, train as trainmod

ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--plugin", required=True); ap.add_argument("--base", required=True)
a = ap.parse_args(); dev = torch.device("cuda:0"); torch.manual_seed(0)
def ok(c, m):
    print(("PASS " if c else "FAIL ") + m, flush=True)
    if not c: sys.exit("SCHEME R SMOKE FAILED")
ck = lambda d: os.path.join(d, sorted(f for f in os.listdir(d) if f.startswith("model_") and f.endswith(".pth"))[-1])

# ---- config exactly as train.py builds it: GlobalConfig defaults + the args garage_train_rank.sh passes + EXTRA ----
cfg = GlobalConfig()
cfg.initialize(root_dir=[a.root], setting="all", use_v2x=1, v2x_rate_dropout=1, v2x_p_zero=0.25, image_architecture="regnety_032",
               lidar_architecture="regnety_032", use_controller_input_prediction=1, v2x_teacher_plugin=a.plugin, v2x_teacher_base=a.base,
               v2x_kd_T=2.0, v2x_kd_w=1.0)
cfg.augment_percentage = 0.0; cfg.use_color_aug = 0   # deterministic batch for the checks; training keeps augmentation
w0 = dict(cfg.detailed_loss_weights); factor = 1.0 / sum(w0.values())
for k in w0: cfg.detailed_loss_weights[k] = w0[k] * factor
ref_ts_w = cfg.detailed_loss_weights["loss_target_speed"]
for k in ("loss_kd_tok_ts", "loss_kd_tok_ck", "loss_kd_zero_ts", "loss_kd_zero_ck"): cfg.detailed_loss_weights[k] = 1.0 * factor
ok(abs(cfg.detailed_loss_weights["loss_target_speed"] - ref_ts_w) < 1e-12 and abs(cfg.detailed_loss_weights["loss_kd_zero_ts"] - ref_ts_w) < 1e-12,
   f"KD weights added after normalisation: existing weights unchanged ({ref_ts_w:.4f}), KD terms on the same scale")

# ---- student = co-trained plug-in, strict ----
student = LidarCenterNet(cfg); student.load_state_dict(torch.load(ck(a.plugin), map_location="cpu"), strict=True); student.to(dev)
ok(True, f"student built from GlobalConfig + train args strict-loads the plug-in checkpoint {ck(a.plugin)} (architecture identical)")
class W(torch.nn.Module):
    def __init__(s, m): super().__init__(); s.module = m
    def forward(s, *x, **k): return s.module(*x, **k)
model = W(student)

ds = datamod.CARLA_Data(root=cfg.data_roots, config=cfg, estimate_class_distributions=False, estimate_sem_distribution=False, shared_dict=None, rank=0)
idx = list(range(0, len(ds), max(1, len(ds) // 8)))[:8]
batch = next(iter(torch.utils.data.DataLoader(torch.utils.data.Subset(ds, idx), batch_size=8, shuffle=False, num_workers=0)))
ok(float(batch["coop_mask"].sum(1).min()) > 0, f"batch: every frame has vehicles (tokens per frame {batch['coop_mask'].sum(1).tolist()})")

args = types.SimpleNamespace(logdir=tempfile.mkdtemp(), save_every=2, backbone="transFuser")
eng = trainmod.Engine(model=model, optimizer=None, dataloader_train=[0], dataloader_val=None, args=args, config=cfg, writer=None, device=dev,
                      scheduler=None, scaler=None)
# teachers: same loading code as train.py main()
eng.teachers = {}
for role, run_dir in (("plugin", a.plugin), ("base", a.base)):
    t_cfg = GlobalConfig(); t_cfg.__dict__.update(jsonpickle.decode(open(os.path.join(run_dir, "config.json")).read()).__dict__)
    t_net = LidarCenterNet(t_cfg); t_net.load_state_dict(torch.load(ck(run_dir), map_location="cpu"), strict=True)
    t_cfg.use_semantic = t_cfg.use_bev_semantic = t_cfg.use_depth = t_cfg.detect_boxes = 0
    t_net.to(dev).eval().requires_grad_(False); eng.teachers[role] = t_net
ok(not any(p.requires_grad for t in eng.teachers.values() for p in t.parameters()), "teachers strict-loaded and frozen")

def run(p_zero, train_mode, amp=True):
    cfg.v2x_p_zero = p_zero; student.train(train_mode)
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=bool(cfg.use_amp) and amp):
        losses, _ = eng.load_data_compute_loss(batch, validation=False)
    return losses

# 2. wiring: student == plug-in teacher in eval mode -> KD to the plug-in is exactly zero; KD to the base is not
# fp16 autocast makes the student (full batch) and the teacher (its subset, a different batch size -> different cuDNN
# algorithms) differ at the 1e-4 m level; the wiring check therefore runs in fp32, the fp16 value is printed as the noise floor
torch.manual_seed(1); L = run(0.0, False, amp=False); torch.manual_seed(1); L16 = run(0.0, False, amp=True)
print(f"eval mode, p_zero 0: fp32 kd_tok_ts {float(L['loss_kd_tok_ts']):.2e} kd_tok_ck {float(L['loss_kd_tok_ck']):.2e} m | fp16 (noise floor) kd_tok_ts {float(L16['loss_kd_tok_ts']):.2e} kd_tok_ck {float(L16['loss_kd_tok_ck']):.2e} m")
ok(float(L["loss_kd_tok_ts"]) < 1e-6 and float(L["loss_kd_tok_ck"]) < 1e-5, "at init the student IS the plug-in teacher: KD-to-plug-in ~ 0 in fp32 on token samples (same gated tokens reach both)")
ok(float(L16["loss_kd_tok_ck"]) < 0.1 * 0.022, "fp16 noise floor of the checkpoint KD is < 10% of the plug-in's rate-0 distance to the base (0.022 m)")
if float(L["loss_kd_zero_ts"]) == 0.0: ok(float(L["loss_kd_zero_ck"]) == 0.0, "no zero-token samples -> KD-to-base exactly 0")
else: print("NOTE random gating left a frame without tokens at p_zero 0; its KD-to-base is non-zero, as designed")
# 1. keys / finiteness (a train-mode forward updates the BatchNorm running statistics, so it must come AFTER the identity check)
L = run(0.25, True)
ok(all(k in L for k in ("loss_kd_tok_ts", "loss_kd_tok_ck", "loss_kd_zero_ts", "loss_kd_zero_ck")), f"KD keys present: {sorted(k for k in L if 'kd' in k)}")
ok(all(k in cfg.detailed_loss_weights for k in L), "every loss key has a weight (the train loop would KeyError otherwise)")
ok(all(torch.isfinite(torch.as_tensor(v)).all() for v in L.values()), "all losses finite")
L = run(1.0, False)
print(f"eval mode, p_zero 1 (all zero-token): kd_zero_ts {float(L['loss_kd_zero_ts']):.4f} kd_zero_ck {float(L['loss_kd_zero_ck']):.4f} m | loss_target_speed {float(L['loss_target_speed']):.2e} loss_checkpoint {float(L['loss_checkpoint']):.2e}")
ok(float(L["loss_kd_zero_ts"]) > 1e-3 or float(L["loss_kd_zero_ck"]) > 1e-3, "the plug-in at rate 0 differs from the base: KD-to-base > 0 (there is something to learn)")
ok(float(L["loss_target_speed"]) == 0.0 and float(L["loss_checkpoint"]) == 0.0, "zero-token samples carry NO imitation planning loss (supervised by the base only)")
ok(float(L["loss_kd_tok_ts"]) == 0.0, "no token samples -> KD-to-plug-in exactly 0")
ok(float(L["loss_semantic"]) > 0 and float(L["loss_bev_semantic"]) > 0, "perception losses still active on zero-token samples")
# 3. gradient: KD-to-base reaches the shared decoder and the backbone; the teachers get none
student.train(); student.zero_grad(set_to_none=True)
L = run(1.0, True); (cfg.detailed_loss_weights["loss_kd_zero_ts"] * L["loss_kd_zero_ts"] + cfg.detailed_loss_weights["loss_kd_zero_ck"] * L["loss_kd_zero_ck"]).backward()
gd = {n.split(".")[0]: float(p.grad.abs().sum()) for n, p in student.named_parameters() if p.grad is not None}
print("grad mass by top-level module:", {k: f"{v:.2e}" for k, v in gd.items() if v > 0})
ok(gd.get("join", 0) > 0 and gd.get("backbone", 0) > 0, "KD-to-base gradient reaches the shared decoder (join) and the backbone")
ok(all(p.grad is None for t in eng.teachers.values() for p in t.parameters()), "no gradient in the teachers")
# 4. a few steps on zero-token samples move the student towards the base
# (BatchNorm frozen during these steps: train-mode BN on 8 frames would move the running statistics and confound the check;
#  the cuDNN GRU needs train mode for backward and has no dropout, so only it is switched)
opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=1e-4)
def run_frozen_bn(p_zero):
    cfg.v2x_p_zero = p_zero; student.eval(); student.checkpoint_decoder.train()
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=bool(cfg.use_amp)):
        losses, _ = eng.load_data_compute_loss(batch, validation=False)
    return losses
traj = []
for step in range(15):
    opt.zero_grad(set_to_none=True); L = run_frozen_bn(1.0); kk = L["loss_kd_zero_ts"] + L["loss_kd_zero_ck"]; traj.append(float(kk)); kk.backward(); opt.step()
L1 = run(1.0, False); k1 = float(L1["loss_kd_zero_ts"]) + float(L1["loss_kd_zero_ck"])
print("KD-to-base trajectory (BN frozen):", [round(x, 4) for x in traj], "-> eval after", round(k1, 4))
ok(k1 < 0.7 * traj[0], f"15 steps of KD-to-base on this batch: rate-0 distance to the base {traj[0]:.4f} -> {k1:.4f}")
L2 = run(0.0, False)
print(f"after those steps, token samples: kd_tok_ts {float(L2['loss_kd_tok_ts']):.4f} kd_tok_ck {float(L2['loss_kd_tok_ck']):.4f} (pulling rate 0 to the base moved the rate-1 output too -- what the plug-in anchor is there to resist)")
# 5. monitoring checkpoints: save_every 2 keeps epochs 1, 3, ... (cur_epoch % 2 == 0 keeps cur_epoch-1)
saved = []
real_save = torch.save; torch.save = lambda obj, f: open(f, "w").close()
eng.optimizer = types.SimpleNamespace(state_dict=lambda: {}); eng.scaler = types.SimpleNamespace(state_dict=lambda: {}); eng.scheduler = types.SimpleNamespace(state_dict=lambda: {})
cfg.cosine_t0 = 10; cfg.cosine_t_mult = 2
for e in range(6): eng.cur_epoch = e; eng.save()
torch.save = real_save
left = sorted(os.path.basename(f) for f in glob.glob(args.logdir + "/model_*.pth")); print("model files kept after epochs 0..5 with save_every 2:", left)
ok(left == ["model_0001.pth", "model_0003.pth", "model_0005.pth"], "monitoring checkpoints kept every 2 epochs plus the latest")
print("SCHEME R SMOKE PASSED")
