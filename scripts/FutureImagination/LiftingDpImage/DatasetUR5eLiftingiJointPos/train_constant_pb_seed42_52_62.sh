#!/usr/bin/env bash
set -e

# Run from the repository root.
#
# Prepare the oracle constant-PB dataset from the new DatasetMujocoUR5eLiftingi
# training split. The copied material_property trajectories are replaced by
# the trained PB of the corresponding object.
mkdir -p robo_manip_baselines/dataset/DatasetMujocoUR5eLiftingi_JointPos_ConstantPB
cp -aL \
  robo_manip_baselines/dataset/DatasetMujocoUR5eLiftingi/training/. \
  robo_manip_baselines/dataset/DatasetMujocoUR5eLiftingi_JointPos_ConstantPB/

python robo_manip_baselines/policy/wrench_predictor4_online/AddConstantPbToDataset.py \
  robo_manip_baselines/dataset/DatasetMujocoUR5eLiftingi_JointPos_ConstantPB \
  --checkpoint robo_manip_baselines/checkpoint/WrenchPredictor4/DatasetMujocoUR5eLiftingi_JointPos/policy_best.ckpt \
  --overwrite

# Train seed 42.
python robo_manip_baselines/bin/Train.py DiffusionPolicy \
  --dataset_dir robo_manip_baselines/dataset/DatasetMujocoUR5eLiftingi_JointPos_ConstantPB \
  --checkpoint_dir robo_manip_baselines/checkpoint/DiffusionPolicy/DatasetMujocoUR5eLiftingi_JointPos_ConstantPB_seed42 \
  --camera_names \
  --state_keys measured_joint_pos measured_tblock_pose material_property \
  --action_keys command_joint_pos \
  --train_ratio 1.0 \
  --val_ratio 0.01

# Train seed 52.
python robo_manip_baselines/bin/Train.py DiffusionPolicy \
  --dataset_dir robo_manip_baselines/dataset/DatasetMujocoUR5eLiftingi_JointPos_ConstantPB \
  --checkpoint_dir robo_manip_baselines/checkpoint/DiffusionPolicy/DatasetMujocoUR5eLiftingi_JointPos_ConstantPB_seed52 \
  --camera_names \
  --state_keys measured_joint_pos measured_tblock_pose material_property \
  --action_keys command_joint_pos \
  --train_ratio 1.0 \
  --val_ratio 0.01 \
  --seed 52

# Train seed 62.
python robo_manip_baselines/bin/Train.py DiffusionPolicy \
  --dataset_dir robo_manip_baselines/dataset/DatasetMujocoUR5eLiftingi_JointPos_ConstantPB \
  --checkpoint_dir robo_manip_baselines/checkpoint/DiffusionPolicy/DatasetMujocoUR5eLiftingi_JointPos_ConstantPB_seed62 \
  --camera_names \
  --state_keys measured_joint_pos measured_tblock_pose material_property \
  --action_keys command_joint_pos \
  --train_ratio 1.0 \
  --val_ratio 0.01 \
  --seed 62
