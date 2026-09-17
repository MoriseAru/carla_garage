#!/bin/bash
# offline rate sweep on UN-augmented frames for one run:  bash tools/v2x_adapter_eval_noaug.sh <ID>
ID=$1; RUNS=/work/gn21/n21001/carla_garage_runs; BASE=$RUNS/tfpp_base_000; ROOT=/work/gn21/n21001/carla_garage_data_root
echo "=== offline sweep (no aug) $ID $(date)"
python -u tools/v2x_offline_rate_sweep.py --root $ROOT --v2x $RUNS/$ID --base $BASE --n 3000 --no_aug 1 && cp /work/gn21/n21001/V2XState_Real/tmp/v2x_offline_rate_sweep.json $RUNS/$ID/offline_sweep_noaug.json && echo "SWEEP_SAVED $ID"
