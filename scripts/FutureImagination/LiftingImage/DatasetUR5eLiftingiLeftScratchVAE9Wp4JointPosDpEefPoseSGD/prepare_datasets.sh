#!/usr/bin/env bash
set -e

# Add MomentumSGD online-PB labels to a copy of the new dataset's training split.
source_dataset=robo_manip_baselines/dataset/DatasetMujocoUR5eLiftingi/training
online_dataset=robo_manip_baselines/dataset/DatasetMujocoUR5eLiftingi_LeftScratchVAE9_Wp4JointPos_DpEefPose_SGDOnlinePB_lr15e3
wp4_checkpoint=robo_manip_baselines/checkpoint/WrenchPredictor4/DatasetMujocoUR5eLiftingi_left_image_vae_9_joint_pos/policy_best.ckpt

mkdir -p "$online_dataset"
cp -aL "$source_dataset/." "$online_dataset/"

python robo_manip_baselines/policy/wrench_predictor4_online/AddOnlinePbToDataset.py \
  "$online_dataset" \
  0 \
  --checkpoint "$wp4_checkpoint" \
  --optimizer sgd \
  --momentum 0.9 \
  --lr 1.5e-2 \
  --wrench_loss_weight 0.0 \
  --overwrite
