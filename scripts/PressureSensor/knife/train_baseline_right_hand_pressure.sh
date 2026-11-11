#!/usr/bin/env bash
set -e

# Train front-image Diffusion Policy with wrench and without PB input.
dataset_dir=robo_manip_baselines/dataset/Datasets/1111_knife/
checkpoint_prefix=robo_manip_baselines/checkpoint/DiffusionPolicy/1111_knife/

python robo_manip_baselines/bin/Train.py DiffusionPolicy \
  --dataset_dir "$dataset_dir" \
  --checkpoint_dir "${checkpoint_prefix}_seed42" \
  --camera_names right hand \
  --state_keys measured_eef_pose measured_eef_wrench main_pressure_sensor \
  --action_keys command_eef_pose \
  --train_ratio 1.0 \
  --val_ratio 0.01 \
  --seed 42
