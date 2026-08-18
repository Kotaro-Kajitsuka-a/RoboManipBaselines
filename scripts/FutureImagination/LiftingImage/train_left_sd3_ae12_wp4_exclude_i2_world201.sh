#!/usr/bin/env bash
set -e

# Ablation: keep the frozen left SD3-AE12 features and exclude only I2 world201.
python robo_manip_baselines/bin/Train.py WrenchPredictor4 \
  --dataset_dir robo_manip_baselines/dataset/LiftingAB_B_only \
  --val_dataset_dir robo_manip_baselines/dataset/LiftingAB_B_only_Validation \
  --checkpoint_dir robo_manip_baselines/checkpoint/WrenchPredictor4/LiftingAB_B_only_left_sd3_ae_12_exclude_i2_world201 \
  --camera_names \
  --state_keys measured_eef_pose measured_gripper_joint_pos \
  --action_keys command_eef_pose command_gripper_joint_pos \
  --image_feature_key sd3_vae_left_ae_12 \
  --wrench_source_key measured_eef_wrench \
  --pb_dim 1 \
  --wrench_loss_weight 0.1 \
  --skip_wrench_preprocessing \
  --exclude_rmb_names MujocoUR5eLiftingi_I2_world201_001.rmb \
  --num_epochs 500
