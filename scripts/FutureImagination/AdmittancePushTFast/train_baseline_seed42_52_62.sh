#!/usr/bin/env bash
set -e

# Run from the repository root.

for train_seed in 42 52 62; do
  python robo_manip_baselines/bin/Train.py DiffusionPolicy \
    --dataset_dir robo_manip_baselines/dataset/DatasetMujocoXarm7AdmittancePushtFast \
    --checkpoint_dir "robo_manip_baselines/checkpoint/DiffusionPolicy/AdmittancePushTFast/baseline_seed${train_seed}" \
    --camera_names \
    --state_keys measured_eef_pose measured_tblock_pose \
    --action_keys command_eef_pose \
    --train_ratio 1.0 \
    --val_ratio 0.01 \
    --seed "$train_seed"
done
