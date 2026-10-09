#!/bin/bash
RUNS=/work/gn21/n21001/carla_garage_runs
python -u tools/v2x_attention_mass.py --root /work/gn21/n21001/carla_garage_data_root --n 2000 --rates "1.0" --out /work/gn21/n21001/V2XState_Real/tmp/v2x_attention_mass_d1t3.json --runs d1s0=$RUNS/tfpp_v2xad_d1_000 t3s0=$RUNS/tfpp_v2xad_t3_000 t1s0=$RUNS/tfpp_v2xad_t1_000
