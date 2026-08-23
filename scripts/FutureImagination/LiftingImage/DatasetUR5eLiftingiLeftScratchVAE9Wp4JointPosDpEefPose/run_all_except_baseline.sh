#!/usr/bin/env bash
set -e

# Reuse the existing Left VAE, JointPos WP4, and EEF-pose no-PB baseline.
# PB-labeled datasets are recreated under experiment-specific names.
base_dir=scripts/FutureImagination/LiftingImage/DatasetUR5eLiftingiLeftScratchVAE9Wp4JointPosDpEefPose
wp4_checkpoint=robo_manip_baselines/checkpoint/WrenchPredictor4/DatasetMujocoUR5eLiftingi_left_image_vae_9_joint_pos/policy_best.ckpt
image_vae_checkpoint=robo_manip_baselines/checkpoint/ImageVAE/DatasetMujocoUR5eLiftingi_left_9/final_model

test -f "$wp4_checkpoint"
test -d "$image_vae_checkpoint"

bash "$base_dir/prepare_datasets.sh"
bash "$base_dir/train_DP_online_pb_seed42_52_62.sh"
bash "$base_dir/train_constant_pb_seed42_52_62.sh"
bash "$base_dir/rollout_DP_online_pb_seed42_52_62.sh"
bash "$base_dir/rollout_constant_pb_seed42_52_62.sh"
