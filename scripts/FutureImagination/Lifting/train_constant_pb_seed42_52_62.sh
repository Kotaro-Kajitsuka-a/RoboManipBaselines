#!/usr/bin/env bash
set -e

# Run from the repository root.
#
# Prepare the oracle constant-PB dataset from the same 75 episodes used by the
# proposed online-PB Diffusion Policy. The copied material_property trajectories
# are replaced by the trained PB of the corresponding object.
mkdir -p robo_manip_baselines/dataset/LiftingAB_B_only_ConstantPB
cp -a \
  robo_manip_baselines/dataset/LiftingAB_B_only/. \
  robo_manip_baselines/dataset/LiftingAB_B_only_ConstantPB/

python robo_manip_baselines/policy/wrench_predictor4_online/AddConstantPbToDataset.py \
  robo_manip_baselines/dataset/LiftingAB_B_only_ConstantPB \
  --checkpoint robo_manip_baselines/checkpoint/WrenchPredictor4/LiftingAB_B_only/policy_best.ckpt \
  --overwrite

# Train seed 42.
python robo_manip_baselines/bin/Train.py DiffusionPolicy \
  --dataset_dir robo_manip_baselines/dataset/LiftingAB_B_only_ConstantPB \
  --checkpoint_dir robo_manip_baselines/checkpoint/DiffusionPolicy/LiftingAB_B_only_ConstantPB_seed42 \
  --camera_names \
  --state_keys measured_eef_pose measured_gripper_joint_pos measured_tblock_pose material_property \
  --action_keys command_eef_pose command_gripper_joint_pos \
  --train_ratio 1.0 \
  --val_ratio 0.01

# Train seed 52.
python robo_manip_baselines/bin/Train.py DiffusionPolicy \
  --dataset_dir robo_manip_baselines/dataset/LiftingAB_B_only_ConstantPB \
  --checkpoint_dir robo_manip_baselines/checkpoint/DiffusionPolicy/LiftingAB_B_only_ConstantPB_seed52 \
  --camera_names \
  --state_keys measured_eef_pose measured_gripper_joint_pos measured_tblock_pose material_property \
  --action_keys command_eef_pose command_gripper_joint_pos \
  --train_ratio 1.0 \
  --val_ratio 0.01 \
  --seed 52

# Train seed 62.
python robo_manip_baselines/bin/Train.py DiffusionPolicy \
  --dataset_dir robo_manip_baselines/dataset/LiftingAB_B_only_ConstantPB \
  --checkpoint_dir robo_manip_baselines/checkpoint/DiffusionPolicy/LiftingAB_B_only_ConstantPB_seed62 \
  --camera_names \
  --state_keys measured_eef_pose measured_gripper_joint_pos measured_tblock_pose material_property \
  --action_keys command_eef_pose command_gripper_joint_pos \
  --train_ratio 1.0 \
  --val_ratio 0.01 \
  --seed 62
