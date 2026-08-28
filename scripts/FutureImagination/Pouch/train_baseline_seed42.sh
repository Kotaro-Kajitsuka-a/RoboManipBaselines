#!/usr/bin/env bash
set -e

# Train right-image Diffusion Policy without PB input using seed 42.
dataset_dir=robo_manip_baselines/dataset/0827_DatasetPouch/training
checkpoint_dir=robo_manip_baselines/checkpoint/DiffusionPolicy/DpImageRight/0827_DatasetPouch_EefPose_Baseline_seed42

python robo_manip_baselines/bin/Train.py DiffusionPolicy \
  --dataset_dir "$dataset_dir" \
  --checkpoint_dir "$checkpoint_dir" \
  --camera_names right \
  --state_keys measured_eef_pose \
  --action_keys command_eef_pose \
  --train_ratio 1.0 \
  --val_ratio 0.01
