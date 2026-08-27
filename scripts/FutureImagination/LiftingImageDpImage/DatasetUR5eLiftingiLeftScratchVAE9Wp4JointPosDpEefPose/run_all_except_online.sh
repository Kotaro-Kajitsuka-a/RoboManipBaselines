#!/usr/bin/env bash
set -e

# Run the image-DP training and rollout pipeline without the Online PB method
# from the repository root. The PB-free baseline is shared by all new-dataset
# EEF-pose methods.
script_dir=scripts/FutureImagination/LiftingImageDpImage/DatasetUR5eLiftingiLeftScratchVAE9Wp4JointPosDpEefPose
baseline_script_dir=scripts/FutureImagination/LiftingImageDpImage/DatasetUR5eLiftingi

bash "$script_dir/train_left_image_vae_wp4.sh"
bash "$script_dir/prepare_datasets.sh"
bash "$baseline_script_dir/train_baseline_seed42_52_62.sh"
bash "$script_dir/train_constant_pb_seed42_52_62.sh"
bash "$baseline_script_dir/rollout_baseline_seed42_52_62.sh"
bash "$script_dir/rollout_constant_pb_seed42_52_62.sh"
