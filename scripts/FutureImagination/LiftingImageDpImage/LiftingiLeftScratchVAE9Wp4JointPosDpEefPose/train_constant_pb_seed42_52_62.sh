#!/usr/bin/env bash
set -e

# Run after prepare_datasets.sh. Train the EEF-pose Diffusion Policy comparison
# with each training object's learned constant PB.
dataset_dir=robo_manip_baselines/dataset/LiftingAB_B_only_LeftScratchVAE9_Wp4JointPos_DpEefPose_ConstantPB
checkpoint_prefix=robo_manip_baselines/checkpoint/DiffusionPolicy/LiftingAB_B_only_LeftScratchVAE9_Wp4JointPos_DpEefPose_ConstantPB

python robo_manip_baselines/bin/Train.py DiffusionPolicy \
  --dataset_dir "$dataset_dir" \
  --checkpoint_dir "${checkpoint_prefix}_seed42" \
   --camera_names left \
    --scheduler ddim \
  --state_keys measured_eef_pose measured_gripper_joint_pos measured_tblock_pose material_property \
  --action_keys command_eef_pose command_gripper_joint_pos \
  --train_ratio 1.0 \
  --val_ratio 0.01

python robo_manip_baselines/bin/Train.py DiffusionPolicy \
  --dataset_dir "$dataset_dir" \
  --checkpoint_dir "${checkpoint_prefix}_seed52" \
   --camera_names left \
    --scheduler ddim \
  --state_keys measured_eef_pose measured_gripper_joint_pos measured_tblock_pose material_property \
  --action_keys command_eef_pose command_gripper_joint_pos \
  --train_ratio 1.0 \
  --val_ratio 0.01 \
  --seed 52

python robo_manip_baselines/bin/Train.py DiffusionPolicy \
  --dataset_dir "$dataset_dir" \
  --checkpoint_dir "${checkpoint_prefix}_seed62" \
   --camera_names left \
    --scheduler ddim \
  --state_keys measured_eef_pose measured_gripper_joint_pos measured_tblock_pose material_property \
  --action_keys command_eef_pose command_gripper_joint_pos \
  --train_ratio 1.0 \
  --val_ratio 0.01 \
  --seed 62
