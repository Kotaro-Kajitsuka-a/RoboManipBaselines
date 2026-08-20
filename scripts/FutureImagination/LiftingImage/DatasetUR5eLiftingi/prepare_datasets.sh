#!/usr/bin/env bash
set -e

# Run from the repository root after train_hand_image_vae_wp4.sh. Keep the
# original DatasetMujocoUR5eLiftingi training split untouched and prepare
# independent copies for the proposed and oracle-PB training methods.
mkdir -p \
  robo_manip_baselines/dataset/DatasetMujocoUR5eLiftingi_Image_OnlinePB \
  robo_manip_baselines/dataset/DatasetMujocoUR5eLiftingi_Image_ConstantPB

cp -aL robo_manip_baselines/dataset/DatasetMujocoUR5eLiftingi/training/. \
  robo_manip_baselines/dataset/DatasetMujocoUR5eLiftingi_Image_OnlinePB/
cp -aL robo_manip_baselines/dataset/DatasetMujocoUR5eLiftingi/training/. \
  robo_manip_baselines/dataset/DatasetMujocoUR5eLiftingi_Image_ConstantPB/

python robo_manip_baselines/policy/wrench_predictor4_online/AddOnlinePbToDataset.py \
  robo_manip_baselines/dataset/DatasetMujocoUR5eLiftingi_Image_OnlinePB \
  0 \
  --checkpoint robo_manip_baselines/checkpoint/WrenchPredictor4/DatasetMujocoUR5eLiftingi_hand_image_vae_9/policy_best.ckpt \
  --lr 6e-3 \
  --wrench_loss_weight 0.0 \
  --overwrite

python robo_manip_baselines/policy/wrench_predictor4_online/AddConstantPbToDataset.py \
  robo_manip_baselines/dataset/DatasetMujocoUR5eLiftingi_Image_ConstantPB \
  --checkpoint robo_manip_baselines/checkpoint/WrenchPredictor4/DatasetMujocoUR5eLiftingi_hand_image_vae_9/policy_best.ckpt \
  --overwrite
