#!/usr/bin/env bash
set -e

# Train the constant-PB comparison from the same copied training episodes used
# by the recomputed online-PB condition.
experiment_suffix=${EXPERIMENT_SUFFIX:-}
dataset_dir="robo_manip_baselines/dataset/DatasetMujocoUR5eLiftingi_LeftScratchVAE9_Wp4JointPos_DpEefPose_ConstantPB${experiment_suffix}"
checkpoint_prefix="robo_manip_baselines/checkpoint/DiffusionPolicy/DatasetMujocoUR5eLiftingi_LeftScratchVAE9_Wp4JointPos_DpEefPose_ConstantPB${experiment_suffix}"

test -d "$dataset_dir"
test ! -e "${checkpoint_prefix}_seed42"
test ! -e "${checkpoint_prefix}_seed52"
test ! -e "${checkpoint_prefix}_seed62"

python robo_manip_baselines/bin/Train.py DiffusionPolicy \
  --dataset_dir "$dataset_dir" \
  --checkpoint_dir "${checkpoint_prefix}_seed42" \
  --camera_names \
  --state_keys measured_eef_pose measured_gripper_joint_pos measured_tblock_pose material_property \
  --action_keys command_eef_pose command_gripper_joint_pos \
  --train_ratio 1.0 \
  --val_ratio 0.01 \
  --seed 42

python robo_manip_baselines/bin/Train.py DiffusionPolicy \
  --dataset_dir "$dataset_dir" \
  --checkpoint_dir "${checkpoint_prefix}_seed52" \
  --camera_names \
  --state_keys measured_eef_pose measured_gripper_joint_pos measured_tblock_pose material_property \
  --action_keys command_eef_pose command_gripper_joint_pos \
  --train_ratio 1.0 \
  --val_ratio 0.01 \
  --seed 52

python robo_manip_baselines/bin/Train.py DiffusionPolicy \
  --dataset_dir "$dataset_dir" \
  --checkpoint_dir "${checkpoint_prefix}_seed62" \
  --camera_names \
  --state_keys measured_eef_pose measured_gripper_joint_pos measured_tblock_pose material_property \
  --action_keys command_eef_pose command_gripper_joint_pos \
  --train_ratio 1.0 \
  --val_ratio 0.01 \
  --seed 62
