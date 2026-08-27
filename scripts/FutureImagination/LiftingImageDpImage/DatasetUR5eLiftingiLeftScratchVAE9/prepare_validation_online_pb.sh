#!/usr/bin/env bash
set -e

# Use the existing left-camera ImageVAE 9D and WP4 policy_best.ckpt.
source_dataset=robo_manip_baselines/dataset/DatasetMujocoUR5eLiftingi/validation
online_dataset=robo_manip_baselines/dataset/DatasetMujocoUR5eLiftingi_LeftScratchVAE9_Validation_OnlinePB
wp4_checkpoint=robo_manip_baselines/checkpoint/WrenchPredictor4/DatasetMujocoUR5eLiftingi_left_image_vae_9_joint_pos/policy_best.ckpt

mkdir -p "$online_dataset"
cp -aL "$source_dataset/." "$online_dataset/"

python robo_manip_baselines/policy/wrench_predictor4_online/AddOnlinePbToDataset.py \
  "$online_dataset" \
  0 \
  --checkpoint "$wp4_checkpoint" \
  --lr 6e-3 \
  --wrench_loss_weight 0.0 \
  --overwrite

python robo_manip_baselines/misc/futureimagination/PlotOnlinePbDataset.py \
  "$online_dataset/WrenchPredObject0"

python robo_manip_baselines/misc/futureimagination/PlotOnlinePbDataset.py \
  "$online_dataset/WrenchPredObject1"

python robo_manip_baselines/misc/futureimagination/PlotOnlinePbDataset.py \
  "$online_dataset/WrenchPredObject2"
