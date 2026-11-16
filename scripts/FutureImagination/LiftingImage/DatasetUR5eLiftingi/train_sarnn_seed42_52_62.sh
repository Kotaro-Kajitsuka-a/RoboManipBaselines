#!/usr/bin/env bash
set -e

# Run from the repository root.
# Train SARNN with the left camera and without material_property/PB input.
# State uses EEF pose plus the gripper joint position commands.

# Train seed 42.
python robo_manip_baselines/bin/Train.py Sarnn \
  --dataset_dir robo_manip_baselines/dataset/DatasetMujocoUR5eLiftingi/training \
  --checkpoint_dir robo_manip_baselines/checkpoint/Sarnn/DatasetMujocoUR5eLiftingi_Left_EefPose_seed42 \
  --camera_names left \
  --image_crop_size_list 480 480 \
  --state_keys command_eef_pose command_gripper_joint_pos \
  --train_ratio 1.0 \
  --val_ratio 0.01

# Train seed 52.
python robo_manip_baselines/bin/Train.py Sarnn \
  --dataset_dir robo_manip_baselines/dataset/DatasetMujocoUR5eLiftingi/training \
  --checkpoint_dir robo_manip_baselines/checkpoint/Sarnn/DatasetMujocoUR5eLiftingi_Left_EefPose_seed52 \
  --camera_names left \
  --image_crop_size_list 480 480 \
  --state_keys command_eef_pose command_gripper_joint_pos \
  --train_ratio 1.0 \
  --val_ratio 0.01 \
  --seed 52

# Train seed 62.
python robo_manip_baselines/bin/Train.py Sarnn \
  --dataset_dir robo_manip_baselines/dataset/DatasetMujocoUR5eLiftingi/training \
  --checkpoint_dir robo_manip_baselines/checkpoint/Sarnn/DatasetMujocoUR5eLiftingi_Left_EefPose_seed62 \
  --camera_names left \
  --image_crop_size_list 480 480 \
  --state_keys command_eef_pose command_gripper_joint_pos \
  --train_ratio 1.0 \
  --val_ratio 0.01 \
  --seed 62
