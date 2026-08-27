#!/usr/bin/env bash
set -e

# Run the legacy-dataset image-DP baseline, online-PB, and constant-PB
# experiments with the existing hand-camera VAE 9D and WP4 checkpoints.
script_dir=scripts/FutureImagination/LiftingImageDpImage

bash "$script_dir/prepare_datasets.sh"
bash "$script_dir/train_baseline_seed42_52_62.sh"
bash "$script_dir/train_DP_online_pb_seed42_52_62.sh"
bash "$script_dir/train_constant_pb_seed42_52_62.sh"
bash "$script_dir/prepare_validation_online_pb.sh"
bash "$script_dir/rollout_baseline_seed42_52_62.sh"
bash "$script_dir/rollout_DP_online_pb_seed42_52_62.sh"
bash "$script_dir/rollout_constant_pb_seed42_52_62.sh"
