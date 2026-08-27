#!/usr/bin/env bash
set -e

# Train image-based Diffusion Policy without material_property/PB input on the
# same legacy LiftingAB_B_only dataset used by the PB methods.
dataset_dir=robo_manip_baselines/dataset/LiftingAB_B_only
checkpoint_prefix=robo_manip_baselines/checkpoint/DiffusionPolicy/DpImageLeft/LiftingAB_B_only_Baseline

python robo_manip_baselines/bin/Train.py DiffusionPolicy \
  --dataset_dir "$dataset_dir" \
  --checkpoint_dir "${checkpoint_prefix}_seed42" \
  --camera_names left \
  --scheduler ddim \
  --state_keys measured_eef_pose measured_gripper_joint_pos measured_tblock_pose \
  --action_keys command_eef_pose command_gripper_joint_pos \
  --train_ratio 1.0 \
  --val_ratio 0.01

python robo_manip_baselines/bin/Train.py DiffusionPolicy \
  --dataset_dir "$dataset_dir" \
  --checkpoint_dir "${checkpoint_prefix}_seed52" \
  --camera_names left \
  --scheduler ddim \
  --state_keys measured_eef_pose measured_gripper_joint_pos measured_tblock_pose \
  --action_keys command_eef_pose command_gripper_joint_pos \
  --train_ratio 1.0 \
  --val_ratio 0.01 \
  --seed 52

python robo_manip_baselines/bin/Train.py DiffusionPolicy \
  --dataset_dir "$dataset_dir" \
  --checkpoint_dir "${checkpoint_prefix}_seed62" \
  --camera_names left \
  --scheduler ddim \
  --state_keys measured_eef_pose measured_gripper_joint_pos measured_tblock_pose \
  --action_keys command_eef_pose command_gripper_joint_pos \
  --train_ratio 1.0 \
  --val_ratio 0.01 \
  --seed 62
