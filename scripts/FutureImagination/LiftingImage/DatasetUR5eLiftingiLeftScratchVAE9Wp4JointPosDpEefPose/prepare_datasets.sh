#!/usr/bin/env bash
set -e

# The Left VAE and JointPos WP4 already exist. Recreate only the PB labels used
# to train the new EEF-pose Diffusion Policies.
source_dataset=robo_manip_baselines/dataset/DatasetMujocoUR5eLiftingi/training
online_dataset=robo_manip_baselines/dataset/DatasetMujocoUR5eLiftingi_LeftScratchVAE9_Wp4JointPos_DpEefPose_AdamOnlinePB
constant_dataset=robo_manip_baselines/dataset/DatasetMujocoUR5eLiftingi_LeftScratchVAE9_Wp4JointPos_DpEefPose_ConstantPB
wp4_checkpoint=robo_manip_baselines/checkpoint/WrenchPredictor4/DatasetMujocoUR5eLiftingi_left_image_vae_9_joint_pos/policy_best.ckpt

test -d "$source_dataset"
test -f "$wp4_checkpoint"
test ! -e "$online_dataset"
test ! -e "$constant_dataset"

mkdir -p "$online_dataset" "$constant_dataset"
cp -aL "$source_dataset"/. "$online_dataset"/
cp -aL "$source_dataset"/. "$constant_dataset"/

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
