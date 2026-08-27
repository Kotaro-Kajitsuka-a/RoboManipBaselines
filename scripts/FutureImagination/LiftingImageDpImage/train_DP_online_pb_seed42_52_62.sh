#!/usr/bin/env bash
set -e

# State-based Diffusion Policy trained with causal PB trajectories identified
# from hand-camera VAE 9D prediction errors at learning rate 6e-3.
python robo_manip_baselines/bin/Train.py DiffusionPolicy \
  --dataset_dir robo_manip_baselines/dataset/LiftingImageAB_B_only_OnlinePB \
  --checkpoint_dir robo_manip_baselines/checkpoint/DiffusionPolicy/LiftingImageAB_B_only_Online_seed42 \
   --camera_names left \
    --scheduler ddim \
  --state_keys measured_eef_pose measured_gripper_joint_pos measured_tblock_pose material_property \
  --action_keys command_eef_pose command_gripper_joint_pos \
  --train_ratio 1.0 \
  --val_ratio 0.01

python robo_manip_baselines/bin/Train.py DiffusionPolicy \
  --dataset_dir robo_manip_baselines/dataset/LiftingImageAB_B_only_OnlinePB \
  --checkpoint_dir robo_manip_baselines/checkpoint/DiffusionPolicy/LiftingImageAB_B_only_Online_seed52 \
   --camera_names left \
    --scheduler ddim \
  --state_keys measured_eef_pose measured_gripper_joint_pos measured_tblock_pose material_property \
  --action_keys command_eef_pose command_gripper_joint_pos \
  --train_ratio 1.0 \
  --val_ratio 0.01 \
  --seed 52

python robo_manip_baselines/bin/Train.py DiffusionPolicy \
  --dataset_dir robo_manip_baselines/dataset/LiftingImageAB_B_only_OnlinePB \
  --checkpoint_dir robo_manip_baselines/checkpoint/DiffusionPolicy/LiftingImageAB_B_only_Online_seed62 \
   --camera_names left \
    --scheduler ddim \
  --state_keys measured_eef_pose measured_gripper_joint_pos measured_tblock_pose material_property \
  --action_keys command_eef_pose command_gripper_joint_pos \
  --train_ratio 1.0 \
  --val_ratio 0.01 \
  --seed 62
