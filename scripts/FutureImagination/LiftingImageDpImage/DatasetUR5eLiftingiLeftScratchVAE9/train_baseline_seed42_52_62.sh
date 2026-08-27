#!/usr/bin/env bash
set -e

# Ordinary Diffusion Policy without material_property/PB input.
# This baseline must be retrained because both state and action use joint space.
python robo_manip_baselines/bin/Train.py DiffusionPolicy \
  --dataset_dir robo_manip_baselines/dataset/DatasetMujocoUR5eLiftingi/training \
  --checkpoint_dir robo_manip_baselines/checkpoint/DiffusionPolicy/DpImageLeft/DatasetMujocoUR5eLiftingi_JointPos_Baseline_seed42 \
  --camera_names left \
  --scheduler ddim \
  --state_keys measured_joint_pos measured_tblock_pose \
  --action_keys command_joint_pos \
  --train_ratio 1.0 \
  --val_ratio 0.01

python robo_manip_baselines/bin/Train.py DiffusionPolicy \
  --dataset_dir robo_manip_baselines/dataset/DatasetMujocoUR5eLiftingi/training \
  --checkpoint_dir robo_manip_baselines/checkpoint/DiffusionPolicy/DpImageLeft/DatasetMujocoUR5eLiftingi_JointPos_Baseline_seed52 \
  --camera_names left \
  --scheduler ddim \
  --state_keys measured_joint_pos measured_tblock_pose \
  --action_keys command_joint_pos \
  --train_ratio 1.0 \
  --val_ratio 0.01 \
  --seed 52

python robo_manip_baselines/bin/Train.py DiffusionPolicy \
  --dataset_dir robo_manip_baselines/dataset/DatasetMujocoUR5eLiftingi/training \
  --checkpoint_dir robo_manip_baselines/checkpoint/DiffusionPolicy/DpImageLeft/DatasetMujocoUR5eLiftingi_JointPos_Baseline_seed62 \
  --camera_names left \
  --scheduler ddim \
  --state_keys measured_joint_pos measured_tblock_pose \
  --action_keys command_joint_pos \
  --train_ratio 1.0 \
  --val_ratio 0.01 \
  --seed 62
