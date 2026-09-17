#!/bin/bash
# attention-mass diagnostic over every trained memory-token model (PBS -v cannot carry commas/spaces, hence a script)
python -u tools/v2x_attention_mass.py --root /work/gn21/n21001/carla_garage_data_root --n 2000 --rates "1.0 0.5" \
  --runs orig=/work/gn21/n21001/carla_garage_runs/tfpp_v2x_000 rd=/work/gn21/n21001/carla_garage_runs/tfpp_v2x_rd_000 occ=/work/gn21/n21001/carla_garage_runs/tfpp_v2x_occ_000 dual=/work/gn21/n21001/carla_garage_runs/tfpp_v2x_dual_000 dual2=/work/gn21/n21001/carla_garage_runs/tfpp_v2x_dual2_000 dual3=/work/gn21/n21001/carla_garage_runs/tfpp_v2x_dual3_000
