#!/usr/bin/env bash
set -e

# Run after prepare_datasets.sh.
# DP state/action are both joint-space; material_property is the oracle constant PB.
python robo_manip_baselines/bin/Train.py DiffusionPolicy \
  --dataset_dir robo_manip_baselines/dataset/DatasetMujocoUR5eLiftingi_LeftScratchVAE9_ConstantPB \
  --checkpoint_dir robo_manip_baselines/checkpoint/DiffusionPolicy/DatasetMujocoUR5eLiftingi_LeftScratchVAE9_ConstantPB_seed42 \
  --camera_names \
  --state_keys measured_joint_pos measured_tblock_pose material_property \
  --action_keys command_joint_pos \
  --train_ratio 1.0 \
  --val_ratio 0.01

python robo_manip_baselines/bin/Train.py DiffusionPolicy \
  --dataset_dir robo_manip_baselines/dataset/DatasetMujocoUR5eLiftingi_LeftScratchVAE9_ConstantPB \
  --checkpoint_dir robo_manip_baselines/checkpoint/DiffusionPolicy/DatasetMujocoUR5eLiftingi_LeftScratchVAE9_ConstantPB_seed52 \
  --camera_names \
  --state_keys measured_joint_pos measured_tblock_pose material_property \
  --action_keys command_joint_pos \
  --train_ratio 1.0 \
  --val_ratio 0.01 \
  --seed 52

python robo_manip_baselines/bin/Train.py DiffusionPolicy \
  --dataset_dir robo_manip_baselines/dataset/DatasetMujocoUR5eLiftingi_LeftScratchVAE9_ConstantPB \
  --checkpoint_dir robo_manip_baselines/checkpoint/DiffusionPolicy/DatasetMujocoUR5eLiftingi_LeftScratchVAE9_ConstantPB_seed62 \
  --camera_names \
  --state_keys measured_joint_pos measured_tblock_pose material_property \
  --action_keys command_joint_pos \
  --train_ratio 1.0 \
  --val_ratio 0.01 \
  --seed 62
