#!/usr/bin/env bash
set -e

# Train right-image Diffusion Policy with wrench and without pressure input.
dataset_dir=robo_manip_baselines/dataset/Datasets/1111_knife/
checkpoint_dir=robo_manip_baselines/checkpoint/DiffusionPolicy/1111_knife/right_wrench_no_pressure_seed42

python robo_manip_baselines/bin/Train.py DiffusionPolicy \
  --dataset_dir "$dataset_dir" \
  --checkpoint_dir "$checkpoint_dir" \
  --camera_names right \
  --state_keys measured_eef_pose measured_eef_wrench \
  --action_keys command_eef_pose \
  --train_ratio 1.0 \
  --val_ratio 0.01 \
  --seed 42
