#!/bin/bash
# Does the hidden-vehicle tokens' attention rise or fall when the visible ones are absent? (LOCAL's request, 09/19 05:26 §2)
RUNS=/work/gn21/n21001/carla_garage_runs; ROOT=/work/gn21/n21001/carla_garage_data_root
for T in all hidden visible; do
  echo "=== token subset: $T"
  python -u tools/v2x_attention_mass.py --root $ROOT --n 2000 --rates "1.0" --tokens $T \
    --out /work/gn21/n21001/V2XState_Real/tmp/v2x_attention_mass_$T.json --runs orig=$RUNS/tfpp_v2x_000 || exit 1
done
echo "ATTENTION_HIDVIS_DONE"
