#!/bin/bash
RUNS=/work/gn21/n21001/carla_garage_runs
python -u tools/v2x_attention_mass.py --root /work/gn21/n21001/carla_garage_data_root --n 2000 --rates "1.0 0.5" --out /work/gn21/n21001/V2XState_Real/tmp/v2x_attention_mass_adapters10.json \
  --runs a1k10=$RUNS/tfpp_v2xad_a1k10_000 a1k6=$RUNS/tfpp_v2xad_a1k6_000 a1k10s2=$RUNS/tfpp_v2xad_a1k10_002
