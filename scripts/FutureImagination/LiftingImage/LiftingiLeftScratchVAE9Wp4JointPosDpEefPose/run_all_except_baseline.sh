#!/usr/bin/env bash
set -e

# Run the complete experiment except the baseline from the repository root.
# Reuse the existing left-camera VAE 9D feature and retrain joint-position WP4.
script_dir=scripts/FutureImagination/LiftingImage/LiftingiLeftScratchVAE9Wp4JointPosDpEefPose

bash "$script_dir/check_required_files.sh"
bash "$script_dir/train_wp4.sh"
bash "$script_dir/prepare_datasets.sh"
bash "$script_dir/train_DP_online_pb_seed42_52_62.sh"
bash "$script_dir/train_constant_pb_seed42_52_62.sh"
bash "$script_dir/rollout_DP_online_pb_seed42_52_62.sh"
bash "$script_dir/rollout_constant_pb_seed42_52_62.sh"
