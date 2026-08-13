#!/usr/bin/env bash
set -e

# Run from the repository root.

python robo_manip_baselines/bin/Train.py WrenchPredictor4 \
  --dataset_dir robo_manip_baselines/dataset/LiftingAB_B_only \
  --checkpoint_dir robo_manip_baselines/checkpoint/WrenchPredictor4/LiftingAB_B_only \
  --camera_names \
  --state_keys measured_eef_pose measured_gripper_joint_pos \
  --action_keys command_eef_pose command_gripper_joint_pos \
  --image_feature_key measured_tblock_pose \
  --wrench_source_key measured_eef_wrench \
  --pb_dim 1 \
  --wrench_loss_weight 0.1 \
  --num_epochs 500 \
  --train_ratio 1.0 \
  --val_ratio 0.01
