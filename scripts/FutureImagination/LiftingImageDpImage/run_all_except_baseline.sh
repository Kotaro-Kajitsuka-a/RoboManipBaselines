#!/usr/bin/env bash
set -e

# Train the hand-camera ImageVAE/WP4, then run the image-DP online-PB and
# constant-PB experiments. The already completed baseline is excluded.
script_dir=scripts/FutureImagination/LiftingImageDpImage

bash "$script_dir/train_hand_image_vae_wp4.sh"
bash "$script_dir/prepare_datasets.sh"
bash "$script_dir/train_DP_online_pb_seed42_52_62.sh"
bash "$script_dir/train_constant_pb_seed42_52_62.sh"
bash "$script_dir/prepare_validation_online_pb.sh"
bash "$script_dir/rollout_DP_online_pb_seed42_52_62.sh"
bash "$script_dir/rollout_constant_pb_seed42_52_62.sh"
