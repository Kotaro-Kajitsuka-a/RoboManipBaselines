#!/usr/bin/env bash
set -e

# Run the adopted beta=2 VAE/WP4, PB-free Baseline, and Constant-PB pipeline.
script_dir=scripts/FutureImagination/Pouch
source_training_dataset=robo_manip_baselines/dataset/0827_DatasetPouch/training
constant_training_dataset=robo_manip_baselines/dataset/0827_DatasetPouch_RightVAE9Beta2_Wp4JointPos_ConstantPB/training
wp4_checkpoint=robo_manip_baselines/checkpoint/WrenchPredictor4/0827_DatasetPouch_right_image_vae_9_beta2_joint_pos/policy_best.ckpt

bash "$script_dir/train_right_image_vae_wp4_beta2.sh"

mkdir -p "$constant_training_dataset"
cp -aL "$source_training_dataset/." "$constant_training_dataset/"

python robo_manip_baselines/policy/wrench_predictor4_online/AddConstantPbToDataset.py \
  "$constant_training_dataset" \
  --checkpoint "$wp4_checkpoint" \
  --overwrite

bash "$script_dir/train_baseline_seed42_52_62.sh"
bash "$script_dir/train_constant_pb_seed42_52_62.sh"
