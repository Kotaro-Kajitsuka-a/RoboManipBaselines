#!/usr/bin/env bash
set -e

# Reuse exactly the same M=9, beta=10 trajectories as the previous seed-42
# training. The only experimental change is the additional online_pb_std state.
dataset_dir=robo_manip_baselines/dataset/DatasetMujocoUR5eLiftingi_EefPose_GaussianOnlinePB_beta10
checkpoint_dir=robo_manip_baselines/checkpoint/DiffusionPolicy/DatasetMujocoUR5eLiftingi_EefPose_GaussianOnline_m9_beta10_WithStd_seed42

python -m robo_manip_baselines.bin.Train DiffusionPolicy \
  --dataset_dir "$dataset_dir" \
  --checkpoint_dir "$checkpoint_dir" \
  --camera_names \
  --state_keys measured_eef_pose measured_gripper_joint_pos measured_tblock_pose material_property online_pb_std \
  --action_keys command_eef_pose command_gripper_joint_pos \
  --backbone cnn \
  --scheduler ddim \
  --skip 3 \
  --batch_size 64 \
  --num_workers 2 \
  --num_epochs 500 \
  --lr 1e-4 \
  --train_ratio 1.0 \
  --val_ratio 0.01 \
  --seed 42
