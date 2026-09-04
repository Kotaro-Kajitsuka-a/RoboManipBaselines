#!/usr/bin/env bash
set -e

# Run the VAE/WP4 prerequisites, diagnostic online-PB dataset preparation,
# PB-free baseline, and constant-PB training. Do not train an online-PB DP.
script_dir=scripts/FutureImagination/PushingLine

bash "$script_dir/train_front_image_vae_wp4.sh"
bash "$script_dir/prepare_datasets.sh"
bash "$script_dir/train_baseline_seed42_52_62.sh"
bash "$script_dir/train_constant_pb_seed42_52_62.sh"
