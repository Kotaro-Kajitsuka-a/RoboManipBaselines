#!/usr/bin/env bash
set -e

source_dataset=robo_manip_baselines/dataset/DatasetMujocoUR5eLiftingi/training
online_dataset=robo_manip_baselines/dataset/DatasetMujocoUR5eLiftingi_HandVaeScratch9dim_EefPose_GaussianOnlinePB_beta10_m16_20260821_v1
constant_dataset=robo_manip_baselines/dataset/DatasetMujocoUR5eLiftingi_HandVaeScratch9dim_EefPose_ConstantPB_GaussianStudy_20260821_v1
wp4_checkpoint=robo_manip_baselines/checkpoint/WrenchPredictor4/DatasetMujocoUR5eLiftingi_HandVaeScratch9dim_EefPose_GaussianStudy_20260821_v1/policy_best.ckpt

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
  --update_type gaussian_belief \
  --initial_std 0.25 \
  --num_points 16 \
  --beta 10 \
  --wrench_loss_weight 0.0 \
  --overwrite

python robo_manip_baselines/policy/wrench_predictor4_online/AddConstantPbToDataset.py \
  "$constant_dataset" \
  --checkpoint "$wp4_checkpoint" \
  --overwrite
