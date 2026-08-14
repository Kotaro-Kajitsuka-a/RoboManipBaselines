#!/usr/bin/env bash
set -e

# Run from the repository root after train_wp4.sh.

# Baseline.
python robo_manip_baselines/bin/Train.py DiffusionPolicy \
  --dataset_dir robo_manip_baselines/dataset/DatasetMujocoXarm7AdmittancePushtOracle \
  --checkpoint_dir robo_manip_baselines/checkpoint/DiffusionPolicy/AdmittancePushTOracle/baseline_seed42 \
  --camera_names \
  --state_keys measured_eef_pose measured_tblock_pose \
  --action_keys command_eef_pose \
  --train_ratio 1.0 \
  --val_ratio 0.01 \
  --seed 42

# Proposed online PB.
python robo_manip_baselines/bin/Train.py DiffusionPolicy \
  --dataset_dir robo_manip_baselines/dataset/DatasetMujocoXarm7AdmittancePushtOracle_online_pb2 \
  --checkpoint_dir robo_manip_baselines/checkpoint/DiffusionPolicy/AdmittancePushTOracle/online_seed42 \
  --camera_names \
  --state_keys measured_eef_pose measured_tblock_pose material_property \
  --action_keys command_eef_pose \
  --train_ratio 1.0 \
  --val_ratio 0.01 \
  --seed 42

# Constant PB.
python robo_manip_baselines/bin/Train.py DiffusionPolicy \
  --dataset_dir robo_manip_baselines/dataset/DatasetMujocoXarm7AdmittancePushtOracle_constant_pb2 \
  --checkpoint_dir robo_manip_baselines/checkpoint/DiffusionPolicy/AdmittancePushTOracle/constant_seed42 \
  --camera_names \
  --state_keys measured_eef_pose measured_tblock_pose material_property \
  --action_keys command_eef_pose \
  --train_ratio 1.0 \
  --val_ratio 0.01 \
  --seed 42
