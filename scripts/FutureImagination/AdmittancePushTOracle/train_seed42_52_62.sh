#!/usr/bin/env bash
set -e

# Run from the repository root after train_wp4.sh.

for train_seed in 42 52 62; do
  # Baseline.
  python robo_manip_baselines/bin/Train.py DiffusionPolicy \
    --dataset_dir robo_manip_baselines/dataset/DatasetMujocoXarm7AdmittancePushtOracle \
    --checkpoint_dir "robo_manip_baselines/checkpoint/DiffusionPolicy/AdmittancePushTOracle/baseline_seed${train_seed}" \
    --camera_names \
    --state_keys measured_eef_pose measured_tblock_pose \
    --action_keys command_eef_pose \
    --train_ratio 1.0 \
    --val_ratio 0.01 \
    --seed "$train_seed"

  # Proposed online PB.
  python robo_manip_baselines/bin/Train.py DiffusionPolicy \
    --dataset_dir robo_manip_baselines/dataset/DatasetMujocoXarm7AdmittancePushtOracle_online_pb2 \
    --checkpoint_dir "robo_manip_baselines/checkpoint/DiffusionPolicy/AdmittancePushTOracle/online_seed${train_seed}" \
    --camera_names \
    --state_keys measured_eef_pose measured_tblock_pose material_property \
    --action_keys command_eef_pose \
    --train_ratio 1.0 \
    --val_ratio 0.01 \
    --seed "$train_seed"

  # Constant PB.
  python robo_manip_baselines/bin/Train.py DiffusionPolicy \
    --dataset_dir robo_manip_baselines/dataset/DatasetMujocoXarm7AdmittancePushtOracle_constant_pb2 \
    --checkpoint_dir "robo_manip_baselines/checkpoint/DiffusionPolicy/AdmittancePushTOracle/constant_seed${train_seed}" \
    --camera_names \
    --state_keys measured_eef_pose measured_tblock_pose material_property \
    --action_keys command_eef_pose \
    --train_ratio 1.0 \
    --val_ratio 0.01 \
    --seed "$train_seed"
done

