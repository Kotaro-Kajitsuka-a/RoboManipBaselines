#!/usr/bin/env bash
set -e

# Run from the repository root.
# Train ordinary Diffusion Policy without material_property/PB input.
# State and action use EEF pose plus the gripper joint position.

# Train seed 42.
python robo_manip_baselines/bin/Train.py DiffusionPolicy \
  --dataset_dir robo_manip_baselines/dataset/DatasetMujocoUR5eLiftingi/training \
  --checkpoint_dir robo_manip_baselines/checkpoint/DiffusionPolicy/DpImageLeft/DatasetMujocoUR5eLiftingi_EefPose_Baseline_seed42 \
  --camera_names left \
  --scheduler ddim \
  --state_keys measured_eef_pose measured_gripper_joint_pos measured_tblock_pose \
  --action_keys command_eef_pose command_gripper_joint_pos \
  --train_ratio 1.0 \
  --val_ratio 0.01

# Train seed 52.
python robo_manip_baselines/bin/Train.py DiffusionPolicy \
  --dataset_dir robo_manip_baselines/dataset/DatasetMujocoUR5eLiftingi/training \
  --checkpoint_dir robo_manip_baselines/checkpoint/DiffusionPolicy/DpImageLeft/DatasetMujocoUR5eLiftingi_EefPose_Baseline_seed52 \
  --camera_names left \
  --scheduler ddim \
  --state_keys measured_eef_pose measured_gripper_joint_pos measured_tblock_pose \
  --action_keys command_eef_pose command_gripper_joint_pos \
  --train_ratio 1.0 \
  --val_ratio 0.01 \
  --seed 52

# Train seed 62.
python robo_manip_baselines/bin/Train.py DiffusionPolicy \
  --dataset_dir robo_manip_baselines/dataset/DatasetMujocoUR5eLiftingi/training \
  --checkpoint_dir robo_manip_baselines/checkpoint/DiffusionPolicy/DpImageLeft/DatasetMujocoUR5eLiftingi_EefPose_Baseline_seed62 \
  --camera_names left \
  --scheduler ddim \
  --state_keys measured_eef_pose measured_gripper_joint_pos measured_tblock_pose \
  --action_keys command_eef_pose command_gripper_joint_pos \
  --train_ratio 1.0 \
  --val_ratio 0.01 \
  --seed 62
