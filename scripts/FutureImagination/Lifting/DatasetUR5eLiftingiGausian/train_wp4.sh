#!/usr/bin/env bash
set -e

# Run from the repository root, preferably inside tmux.
checkpoint_dir=robo_manip_baselines/checkpoint/WrenchPredictor4/DatasetMujocoUR5eLiftingi_EefPose_GaussianStudy_20260821_v1

test ! -e "$checkpoint_dir"

python robo_manip_baselines/bin/Train.py WrenchPredictor4 \
  --dataset_dir robo_manip_baselines/dataset/DatasetMujocoUR5eLiftingi/training \
  --val_dataset_dir robo_manip_baselines/dataset/DatasetMujocoUR5eLiftingi/validation \
  --checkpoint_dir "$checkpoint_dir" \
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

python robo_manip_baselines/policy/wrench_predictor4/EvalWrenchPredictor4SweepDir.py \
  "$checkpoint_dir/policy_best.ckpt" \
  robo_manip_baselines/dataset/DatasetMujocoUR5eLiftingi/validation \
  --material_object_ids 0 1 2

echo "WP4 checkpoint: $checkpoint_dir/policy_best.ckpt"
