#!/usr/bin/env bash
set -e

base_dir=scripts/FutureImagination/LiftingImage/DatasetUR5eLiftingiGaussian/LeftVaeScratch9dimAllJointPos
bash "$base_dir/train_left_image_vae_wp4.sh"
bash "$base_dir/prepare_datasets.sh"
bash "$base_dir/train_DP_online_pb_seed42_52_62.sh"
bash "$base_dir/train_constant_pb_seed42_52_62.sh"
bash "$base_dir/rollout_DP_online_pb_seed42_52_62.sh"
bash "$base_dir/rollout_constant_pb_seed42_52_62.sh"
