#!/usr/bin/env bash
set -e

# Run after prepare_datasets.sh. Each episode uses its object's learned PB.
python robo_manip_baselines/bin/Train.py DiffusionPolicy \
  --dataset_dir robo_manip_baselines/dataset/DatasetMujocoUR5eLiftingi_LeftLarge3Z16_Wp4JointPos_DpEefPose_ConstantPB \
  --checkpoint_dir robo_manip_baselines/checkpoint/DiffusionPolicy/DatasetMujocoUR5eLiftingi_LeftLarge3Z16_Wp4JointPos_DpEefPose_ConstantPB_seed42 \
  --camera_names \
  --state_keys measured_eef_pose measured_gripper_joint_pos measured_tblock_pose material_property \
  --action_keys command_eef_pose command_gripper_joint_pos \
  --train_ratio 1.0 \
  --val_ratio 0.01
