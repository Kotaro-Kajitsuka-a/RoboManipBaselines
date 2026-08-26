#!/usr/bin/env bash
set -e

# Run after train_wp4.sh. Keep the source dataset intact while adding PB labels
# to independent copies.
source_dataset=robo_manip_baselines/dataset/LiftingAB_B_only
online_dataset=robo_manip_baselines/dataset/LiftingAB_B_only_LeftScratchVAE9_Wp4JointPos_DpEefPose_AdamOnlinePB
constant_dataset=robo_manip_baselines/dataset/LiftingAB_B_only_LeftScratchVAE9_Wp4JointPos_DpEefPose_ConstantPB
wp4_checkpoint=robo_manip_baselines/checkpoint/WrenchPredictor4/LiftingAB_B_only_left_image_vae_9_joint_pos/policy_best.ckpt

mkdir -p "$online_dataset" "$constant_dataset"
cp -aL "$source_dataset/." "$online_dataset/"
cp -aL "$source_dataset/." "$constant_dataset/"

python robo_manip_baselines/policy/wrench_predictor4_online/AddOnlinePbToDataset.py \
  "$online_dataset" \
  0 \
  --checkpoint "$wp4_checkpoint" \
  --lr 6e-3 \
  --wrench_loss_weight 0.0 \
  --overwrite

python robo_manip_baselines/policy/wrench_predictor4_online/AddConstantPbToDataset.py \
  "$constant_dataset" \
  --checkpoint "$wp4_checkpoint" \
  --overwrite
