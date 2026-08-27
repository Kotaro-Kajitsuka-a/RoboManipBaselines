#!/usr/bin/env bash
set -e

# Run the complete training pipeline except the online-PB method.
script_dir=scripts/FutureImagination/Sorting
source_training_dataset=robo_manip_baselines/dataset/0826_DatasetSorting/training
constant_training_dataset=robo_manip_baselines/dataset/0826_DatasetSorting_FrontLarge3Z16_Wp4JointPos_ConstantPB/training
wp4_checkpoint=robo_manip_baselines/checkpoint/WrenchPredictor4/0826_DatasetSorting_front_image_vae_large3_z16_joint_pos/policy_best.ckpt

bash "$script_dir/train_front_image_vae_large3_z16.sh"
bash "$script_dir/train_baseline_seed42_52_62.sh"

mkdir -p "$constant_training_dataset"
cp -aL "$source_training_dataset/." "$constant_training_dataset/"

python robo_manip_baselines/policy/wrench_predictor4_online/AddConstantPbToDataset.py \
  "$constant_training_dataset" \
  --checkpoint "$wp4_checkpoint" \
  --overwrite

bash "$script_dir/train_constant_pb_seed42_52_62.sh"
