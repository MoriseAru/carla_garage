#!/bin/bash
# attention-mass / residual diagnostic for the three adapter variants:  bash tools/v2x_attention_mass_adapters.sh [suffix]
SUFFIX=${1:-000}; RUNS=/work/gn21/n21001/carla_garage_runs
python -u tools/v2x_attention_mass.py --root /work/gn21/n21001/carla_garage_data_root --n 2000 --rates "1.0 0.5" --out /work/gn21/n21001/V2XState_Real/tmp/v2x_attention_mass_adapters.json \
  --runs a1=$RUNS/tfpp_v2xad_a1_$SUFFIX a2=$RUNS/tfpp_v2xad_a2_$SUFFIX a3=$RUNS/tfpp_v2xad_a3_$SUFFIX
