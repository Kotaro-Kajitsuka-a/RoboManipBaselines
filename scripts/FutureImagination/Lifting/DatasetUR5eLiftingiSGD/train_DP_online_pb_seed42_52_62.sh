#!/usr/bin/env bash
set -e

# Keep the original training split untouched and add the causal MomentumSGD
# online-PB trajectories to an independent copy used by Diffusion Policy.
mkdir -p robo_manip_baselines/dataset/DatasetMujocoUR5eLiftingi_EefPose_SGDOnlinePB_lr8e3
cp -aL \
  robo_manip_baselines/dataset/DatasetMujocoUR5eLiftingi/training/. \
  robo_manip_baselines/dataset/DatasetMujocoUR5eLiftingi_EefPose_SGDOnlinePB_lr8e3/

python robo_manip_baselines/policy/wrench_predictor4_online/AddOnlinePbToDataset.py \
  robo_manip_baselines/dataset/DatasetMujocoUR5eLiftingi_EefPose_SGDOnlinePB_lr8e3 \
  0 \
  --checkpoint robo_manip_baselines/checkpoint/WrenchPredictor4/DatasetMujocoUR5eLiftingi_EefPose/policy_best.ckpt \
  --optimizer sgd \
  --momentum 0.9 \
  --lr 8e-3 \
  --wrench_loss_weight 0.0 \
  --overwrite

# Train seed 42.
python robo_manip_baselines/bin/Train.py DiffusionPolicy \
  --dataset_dir robo_manip_baselines/dataset/DatasetMujocoUR5eLiftingi_EefPose_SGDOnlinePB_lr8e3 \
  --checkpoint_dir robo_manip_baselines/checkpoint/DiffusionPolicy/DatasetMujocoUR5eLiftingi_EefPose_SGDOnline_lr8e3_seed42 \
  --camera_names \
  --state_keys measured_eef_pose measured_gripper_joint_pos measured_tblock_pose material_property \
  --action_keys command_eef_pose command_gripper_joint_pos \
  --train_ratio 1.0 \
  --val_ratio 0.01

# Train seed 52.
python robo_manip_baselines/bin/Train.py DiffusionPolicy \
  --dataset_dir robo_manip_baselines/dataset/DatasetMujocoUR5eLiftingi_EefPose_SGDOnlinePB_lr8e3 \
  --checkpoint_dir robo_manip_baselines/checkpoint/DiffusionPolicy/DatasetMujocoUR5eLiftingi_EefPose_SGDOnline_lr8e3_seed52 \
  --camera_names \
  --state_keys measured_eef_pose measured_gripper_joint_pos measured_tblock_pose material_property \
  --action_keys command_eef_pose command_gripper_joint_pos \
  --train_ratio 1.0 \
  --val_ratio 0.01 \
  --seed 52

# Train seed 62.
python robo_manip_baselines/bin/Train.py DiffusionPolicy \
  --dataset_dir robo_manip_baselines/dataset/DatasetMujocoUR5eLiftingi_EefPose_SGDOnlinePB_lr8e3 \
  --checkpoint_dir robo_manip_baselines/checkpoint/DiffusionPolicy/DatasetMujocoUR5eLiftingi_EefPose_SGDOnline_lr8e3_seed62 \
  --camera_names \
  --state_keys measured_eef_pose measured_gripper_joint_pos measured_tblock_pose material_property \
  --action_keys command_eef_pose command_gripper_joint_pos \
  --train_ratio 1.0 \
  --val_ratio 0.01 \
  --seed 62
