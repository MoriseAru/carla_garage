#!/bin/bash
# Scheme A on the full cache: three token-gating variants of the frozen-base adapter, then the offline rate sweep and the
# attention-mass diagnostic on each. One node, ~1-2 h.   qsub -q short-g -l walltime=04:00:00 -v T=tools/v2x_adapter_train_all.sh train_scripts/garage_g_bash.sh
set -o pipefail
CACHE=${CACHE:-/work/gn21/n21001/carla_garage_cache/base_000}; BASE=/work/gn21/n21001/carla_garage_runs/tfpp_base_000
ROOT=/work/gn21/n21001/carla_garage_data_root; RUNS=/work/gn21/n21001/carla_garage_runs; EPOCHS=${EPOCHS:-10}; SUFFIX=${SUFFIX:-000}
ls $CACHE/shard*_meta.json >/dev/null || { echo "no cache"; exit 1; }
for V in "a1 rate1" "a2 random" "a3 vis"; do
  set -- $V; ID=tfpp_v2xad_${1}_$SUFFIX; echo "=== train $ID (gating $2) $(date)"
  python -u tools/v2x_train_adapter.py --cache $CACHE --base $BASE --id $ID --epochs $EPOCHS --bs 512 --gating $2 --tokens all || exit 1
done
for V in a1 a2 a3; do
  ID=tfpp_v2xad_${V}_$SUFFIX; echo "=== offline sweep $ID $(date)"
  python -u tools/v2x_offline_rate_sweep.py --root $ROOT --v2x $RUNS/$ID --base $BASE --n 3000 || exit 1
  cp /work/gn21/n21001/V2XState_Real/tmp/v2x_offline_rate_sweep.json $RUNS/$ID/offline_sweep.json
done
echo "=== attention mass (adapters) $(date)"
python -u tools/v2x_attention_mass.py --root $ROOT --n 2000 --rates "1.0 0.5" --out /work/gn21/n21001/V2XState_Real/tmp/v2x_attention_mass_adapters.json \
  --runs a1=$RUNS/tfpp_v2xad_a1_$SUFFIX a2=$RUNS/tfpp_v2xad_a2_$SUFFIX a3=$RUNS/tfpp_v2xad_a3_$SUFFIX || exit 1
echo "ADAPTER_ALL_DONE"
