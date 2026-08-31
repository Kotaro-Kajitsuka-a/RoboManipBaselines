#!/usr/bin/env bash
set -e

# Run after prepare_datasets.sh. Each episode uses its beta=2 WP4 causal online PB.
dataset_dir=robo_manip_baselines/dataset/0827_DatasetPouch_RightVAE9Beta2_Wp4JointPos_AdamOnlinePB/training
checkpoint_prefix=robo_manip_baselines/checkpoint/DiffusionPolicy/DpImageRight/0827_DatasetPouch_RightVAE9Beta2_Wp4JointPos_EefPose_AdamOnlinePB

python robo_manip_baselines/bin/Train.py DiffusionPolicy \
  --dataset_dir "$dataset_dir" \
  --checkpoint_dir "${checkpoint_prefix}_seed42" \
  --camera_names right \
  --state_keys measured_eef_pose measured_gripper_joint_pos material_property \
  --action_keys command_eef_pose command_gripper_joint_pos \
  --train_ratio 1.0 \
  --val_ratio 0.01

python robo_manip_baselines/bin/Train.py DiffusionPolicy \
  --dataset_dir "$dataset_dir" \
  --checkpoint_dir "${checkpoint_prefix}_seed52" \
  --camera_names right \
  --state_keys measured_eef_pose measured_gripper_joint_pos material_property \
  --action_keys command_eef_pose command_gripper_joint_pos \
  --train_ratio 1.0 \
  --val_ratio 0.01 \
  --seed 52

python robo_manip_baselines/bin/Train.py DiffusionPolicy \
  --dataset_dir "$dataset_dir" \
  --checkpoint_dir "${checkpoint_prefix}_seed62" \
  --camera_names right \
  --state_keys measured_eef_pose measured_gripper_joint_pos material_property \
  --action_keys command_eef_pose command_gripper_joint_pos \
  --train_ratio 1.0 \
  --val_ratio 0.01 \
  --seed 62
