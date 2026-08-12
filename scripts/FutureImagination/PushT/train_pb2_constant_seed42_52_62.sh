#!/usr/bin/env bash
set -e

# Run from the repository root.

# Train seed 42.
python robo_manip_baselines/bin/Train.py DiffusionPolicy \
  --dataset_dir robo_manip_baselines/dataset/DatasetMujocoXarm7Pusht_constantpb2_wp4seed42_20260810 \
  --checkpoint_dir robo_manip_baselines/checkpoint/DiffusionPolicy/PushTPB2/constant_seed42 \
  --camera_names \
  --state_keys measured_eef_pose measured_tblock_pose material_property \
  --action_keys command_eef_pose \
  --train_ratio 1.0 \
  --val_ratio 0.01 \
  --seed 42

# Train seed 52.
python robo_manip_baselines/bin/Train.py DiffusionPolicy \
  --dataset_dir robo_manip_baselines/dataset/DatasetMujocoXarm7Pusht_constantpb2_wp4seed42_20260810 \
  --checkpoint_dir robo_manip_baselines/checkpoint/DiffusionPolicy/PushTPB2/constant_seed52 \
  --camera_names \
  --state_keys measured_eef_pose measured_tblock_pose material_property \
  --action_keys command_eef_pose \
  --train_ratio 1.0 \
  --val_ratio 0.01 \
  --seed 52

# Train seed 62.
python robo_manip_baselines/bin/Train.py DiffusionPolicy \
  --dataset_dir robo_manip_baselines/dataset/DatasetMujocoXarm7Pusht_constantpb2_wp4seed42_20260810 \
  --checkpoint_dir robo_manip_baselines/checkpoint/DiffusionPolicy/PushTPB2/constant_seed62 \
  --camera_names \
  --state_keys measured_eef_pose measured_tblock_pose material_property \
  --action_keys command_eef_pose \
  --train_ratio 1.0 \
  --val_ratio 0.01 \
  --seed 62
