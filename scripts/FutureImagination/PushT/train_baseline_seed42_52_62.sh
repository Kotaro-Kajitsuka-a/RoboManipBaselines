#!/usr/bin/env bash
set -e

# Run from the repository root.

# Train seed 42.
python robo_manip_baselines/bin/Train.py DiffusionPolicy \
  --dataset_dir robo_manip_baselines/dataset/DatasetMujocoXarm7Pusht \
  --checkpoint_dir robo_manip_baselines/checkpoint/DiffusionPolicy/PushTBaseline/seed42 \
  --camera_names \
  --state_keys measured_eef_pose measured_tblock_pose \
  --action_keys command_eef_pose \
  --train_ratio 1.0 \
  --val_ratio 0.01

# Train seed 52.
python robo_manip_baselines/bin/Train.py DiffusionPolicy \
  --dataset_dir robo_manip_baselines/dataset/DatasetMujocoXarm7Pusht \
  --checkpoint_dir robo_manip_baselines/checkpoint/DiffusionPolicy/PushTBaseline/seed52 \
  --camera_names \
  --state_keys measured_eef_pose measured_tblock_pose \
  --action_keys command_eef_pose \
  --train_ratio 1.0 \
  --val_ratio 0.01 \
  --seed 52

# Train seed 62.
python robo_manip_baselines/bin/Train.py DiffusionPolicy \
  --dataset_dir robo_manip_baselines/dataset/DatasetMujocoXarm7Pusht \
  --checkpoint_dir robo_manip_baselines/checkpoint/DiffusionPolicy/PushTBaseline/seed62 \
  --camera_names \
  --state_keys measured_eef_pose measured_tblock_pose \
  --action_keys command_eef_pose \
  --train_ratio 1.0 \
  --val_ratio 0.01 \
  --seed 62
