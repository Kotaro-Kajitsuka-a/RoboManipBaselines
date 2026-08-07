#!/usr/bin/env bash
set -e

# Add the causal online-PB trajectories used to train Diffusion Policy.
python robo_manip_baselines/policy/wrench_predictor4_online/AddOnlinePbToDataset.py \
  robo_manip_baselines/dataset/LiftingAB_B_only \
  0 \
  --checkpoint robo_manip_baselines/checkpoint/WrenchPredictor4/LiftingAB_B_only/policy_best.ckpt \
  --lr 6e-3 \
  --overwrite

# Train seed 42.
python robo_manip_baselines/bin/Train.py DiffusionPolicy \
  --dataset_dir robo_manip_baselines/dataset/LiftingAB_B_only \
  --checkpoint_dir robo_manip_baselines/checkpoint/DiffusionPolicy/LiftingAB_B_only_Online_seed42 \
  --camera_names \
  --state_keys measured_eef_pose measured_gripper_joint_pos measured_tblock_pose material_property \
  --action_keys command_eef_pose command_gripper_joint_pos \
  --train_ratio 1.0 \
  --val_ratio 0.01

# Train seed 52.
python robo_manip_baselines/bin/Train.py DiffusionPolicy \
  --dataset_dir robo_manip_baselines/dataset/LiftingAB_B_only \
  --checkpoint_dir robo_manip_baselines/checkpoint/DiffusionPolicy/LiftingAB_B_only_Online_seed52 \
  --camera_names \
  --state_keys measured_eef_pose measured_gripper_joint_pos measured_tblock_pose material_property \
  --action_keys command_eef_pose command_gripper_joint_pos \
  --train_ratio 1.0 \
  --val_ratio 0.01 \
  --seed 52

# Train seed 62.
python robo_manip_baselines/bin/Train.py DiffusionPolicy \
  --dataset_dir robo_manip_baselines/dataset/LiftingAB_B_only \
  --checkpoint_dir robo_manip_baselines/checkpoint/DiffusionPolicy/LiftingAB_B_only_Online_seed62 \
  --camera_names \
  --state_keys measured_eef_pose measured_gripper_joint_pos measured_tblock_pose material_property \
  --action_keys command_eef_pose command_gripper_joint_pos \
  --train_ratio 1.0 \
  --val_ratio 0.01 \
  --seed 62
