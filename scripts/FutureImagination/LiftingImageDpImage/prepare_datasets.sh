#!/usr/bin/env bash
set -e

# Run from the repository root. Keep the original 75 episodes untouched and
# prepare independent copies for the proposed and oracle-PB training methods.
mkdir -p \
  robo_manip_baselines/dataset/LiftingImageAB_B_only_OnlinePB \
  robo_manip_baselines/dataset/LiftingImageAB_B_only_ConstantPB

cp -aL robo_manip_baselines/dataset/LiftingAB_B_only/. \
  robo_manip_baselines/dataset/LiftingImageAB_B_only_OnlinePB/
cp -aL robo_manip_baselines/dataset/LiftingAB_B_only/. \
  robo_manip_baselines/dataset/LiftingImageAB_B_only_ConstantPB/

python robo_manip_baselines/policy/wrench_predictor4_online/AddOnlinePbToDataset.py \
  robo_manip_baselines/dataset/LiftingImageAB_B_only_OnlinePB \
  0 \
  --checkpoint robo_manip_baselines/checkpoint/WrenchPredictor4/LiftingAB_B_only_hand_image_vae_9/policy_best.ckpt \
  --lr 6e-3 \
  --wrench_loss_weight 0.0 \
  --overwrite

python robo_manip_baselines/policy/wrench_predictor4_online/AddConstantPbToDataset.py \
  robo_manip_baselines/dataset/LiftingImageAB_B_only_ConstantPB \
  --checkpoint robo_manip_baselines/checkpoint/WrenchPredictor4/LiftingAB_B_only_hand_image_vae_9/policy_best.ckpt \
  --overwrite
