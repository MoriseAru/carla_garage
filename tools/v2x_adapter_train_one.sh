#!/bin/bash
# one scheme-A adapter variant on the full cache (debug-g sized):  bash tools/v2x_adapter_train_one.sh a1 rate1 [epochs] [suffix]
V=$1; GATING=$2; EPOCHS=${3:-10}; SUFFIX=${4:-000}; ID=tfpp_v2xad_${V}_$SUFFIX
CACHE=${CACHE:-/work/gn21/n21001/carla_garage_cache/base_000}; BASE=/work/gn21/n21001/carla_garage_runs/tfpp_base_000
echo "=== train $ID (gating $GATING, $EPOCHS epochs) $(date)"
python -u tools/v2x_train_adapter.py --cache $CACHE --base $BASE --id $ID --epochs $EPOCHS --bs 512 --gating $GATING --tokens all
