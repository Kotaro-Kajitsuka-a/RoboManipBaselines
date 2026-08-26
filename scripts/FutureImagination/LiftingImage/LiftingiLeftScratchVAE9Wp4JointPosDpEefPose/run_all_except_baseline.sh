#!/usr/bin/env bash
set -e

# Run the complete experiment except the baseline from the repository root.
# Train left-camera VAE 9D and joint-position WP4 before the DP experiments.
script_dir=scripts/FutureImagination/LiftingImage/LiftingiLeftScratchVAE9Wp4JointPosDpEefPose

bash "$script_dir/check_required_files.sh"
bash "$script_dir/train_left_image_vae_wp4.sh"
bash "$script_dir/prepare_datasets.sh"
bash "$script_dir/train_DP_online_pb_seed42_52_62.sh"
bash "$script_dir/train_constant_pb_seed42_52_62.sh"
bash "$script_dir/rollout_DP_online_pb_seed42_52_62.sh"
bash "$script_dir/rollout_constant_pb_seed42_52_62.sh"
