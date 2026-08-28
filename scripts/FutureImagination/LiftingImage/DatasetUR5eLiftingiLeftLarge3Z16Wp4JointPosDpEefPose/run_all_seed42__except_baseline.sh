#!/usr/bin/env bash
set -e

# Train and evaluate the seed-42 PB methods without the baseline.
script_dir=scripts/FutureImagination/LiftingImage/DatasetUR5eLiftingiLeftLarge3Z16Wp4JointPosDpEefPose

bash "$script_dir/train_left_image_vae_large3_z16_wp4.sh"
bash "$script_dir/prepare_datasets.sh"
bash "$script_dir/train_constant_pb_seed42.sh"
bash "$script_dir/train_online_pb_seed42.sh"
bash "$script_dir/rollout_seed42.sh" constant_pb
bash "$script_dir/rollout_seed42.sh" online_pb
