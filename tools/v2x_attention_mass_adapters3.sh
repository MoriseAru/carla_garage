#!/bin/bash
RUNS=/work/gn21/n21001/carla_garage_runs
python -u tools/v2x_attention_mass.py --root /work/gn21/n21001/carla_garage_data_root --n 2000 --rates "1.0 0.5" --out /work/gn21/n21001/V2XState_Real/tmp/v2x_attention_mass_adapters3.json \
  --runs a1k=$RUNS/tfpp_v2xad_a1k_000 a1kc=$RUNS/tfpp_v2xad_a1kc_000 a3k=$RUNS/tfpp_v2xad_a3k_000
