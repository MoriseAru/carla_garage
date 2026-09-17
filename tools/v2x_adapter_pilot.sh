#!/bin/bash
# Pilot for scheme A: mini feature cache on the smoke root, then a 3-epoch adapter training on it (structure + numbers sanity).
set -o pipefail
OUT=/work/gn21/n21001/carla_garage_cache/pilot_smoke; rm -rf $OUT; mkdir -p $OUT
python -u tools/v2x_cache_features.py --root /work/gn21/n21001/carla_garage_smoke_root --base /work/gn21/n21001/carla_garage_runs/tfpp_base_000 --out $OUT --shard 0 --nshards 2 --bs 32 --workers 16 --max_frames 1024 || exit 1
python -u tools/v2x_cache_features.py --root /work/gn21/n21001/carla_garage_smoke_root --base /work/gn21/n21001/carla_garage_runs/tfpp_base_000 --out $OUT --shard 1 --nshards 2 --bs 32 --workers 16 --max_frames 1024 || exit 1
ls -la $OUT | awk '{print $5, $9}' | tail -20
python -u tools/v2x_train_adapter.py --cache $OUT --base /work/gn21/n21001/carla_garage_runs/tfpp_base_000 --id pilot_v2xad_a1 --epochs 3 --bs 128 --warmup 20 --gating rate1 --val_frac 0.2 || exit 1
python -u tools/v2x_train_adapter.py --cache $OUT --base /work/gn21/n21001/carla_garage_runs/tfpp_base_000 --id pilot_v2xad_a3 --epochs 2 --bs 128 --warmup 20 --gating vis --val_frac 0.2 || exit 1
python - <<'PY'
import torch, json, sys
sd = torch.load("/work/gn21/n21001/carla_garage_runs/pilot_v2xad_a1/model_0030.pth", map_location="cpu")
ad = [k for k in sd if k.startswith("v2x_adapter.")]; print("exported tensors", len(sd), "adapter", len(ad))
m = json.load(open("/work/gn21/n21001/carla_garage_runs/pilot_v2xad_a1/metrics.json")); h = m["history"]
print("train loss per epoch:", [round(x["train_loss"], 4) for x in h]); assert h[-1]["train_loss"] < h[0]["train_loss"], "loss did not decrease"
print("PILOT_OK")
PY
