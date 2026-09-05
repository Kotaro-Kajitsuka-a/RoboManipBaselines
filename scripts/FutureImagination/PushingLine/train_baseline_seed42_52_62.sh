#!/usr/bin/env bash
set -e

# Train front-image Diffusion Policy with wrench and without PB input.
dataset_dir=robo_manip_baselines/dataset/DatasetPushingLine/training
checkpoint_prefix=robo_manip_baselines/checkpoint/DiffusionPolicy/DpImageFront/DatasetPushingLine_EefPoseWrench_Baseline

python robo_manip_baselines/bin/Train.py DiffusionPolicy \
  --dataset_dir "$dataset_dir" \
  --checkpoint_dir "${checkpoint_prefix}_seed42" \
  --camera_names front \
  --state_keys measured_eef_pose measured_eef_wrench \
  --action_keys command_eef_pose \
  --train_ratio 1.0 \
  --val_ratio 0.01

python robo_manip_baselines/bin/Train.py DiffusionPolicy \
  --dataset_dir "$dataset_dir" \
  --checkpoint_dir "${checkpoint_prefix}_seed52" \
  --camera_names front \
  --state_keys measured_eef_pose measured_eef_wrench \
  --action_keys command_eef_pose \
  --train_ratio 1.0 \
  --val_ratio 0.01 \
  --seed 52

python robo_manip_baselines/bin/Train.py DiffusionPolicy \
  --dataset_dir "$dataset_dir" \
  --checkpoint_dir "${checkpoint_prefix}_seed62" \
  --camera_names front \
  --state_keys measured_eef_pose measured_eef_wrench \
  --action_keys command_eef_pose \
  --train_ratio 1.0 \
  --val_ratio 0.01 \
  --seed 62
