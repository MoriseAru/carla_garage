#!/bin/bash
RUNS=/work/gn21/n21001/carla_garage_runs
python -u tools/v2x_attention_mass.py --root /work/gn21/n21001/carla_garage_data_root --n 2000 --rates "1.0 0.5" --out /work/gn21/n21001/V2XState_Real/tmp/v2x_attention_mass_adapters2.json \
  --runs a1c=$RUNS/tfpp_v2xad_a1c_000 a1n=$RUNS/tfpp_v2xad_a1n_000 a1nc=$RUNS/tfpp_v2xad_a1nc_000
