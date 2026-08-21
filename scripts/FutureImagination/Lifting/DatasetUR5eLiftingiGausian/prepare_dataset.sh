#!/usr/bin/env bash
set -e

# Run from the repository root after train_wp4.sh.
source_dataset_dir=robo_manip_baselines/dataset/DatasetMujocoUR5eLiftingi/training
constant_dataset_dir=robo_manip_baselines/dataset/DatasetMujocoUR5eLiftingi_EefPose_ConstantPB_GaussianStudy_20260821_v1
online_dataset_dir=robo_manip_baselines/dataset/DatasetMujocoUR5eLiftingi_EefPose_GaussianOnlinePB_beta10_m16_20260821_v1
wp4_checkpoint=robo_manip_baselines/checkpoint/WrenchPredictor4/DatasetMujocoUR5eLiftingi_EefPose_GaussianStudy_20260821_v1/policy_best.ckpt

test -d "$source_dataset_dir"
test -f "$wp4_checkpoint"
test ! -e "$constant_dataset_dir"
test ! -e "$online_dataset_dir"

mkdir -p "$constant_dataset_dir"
cp -aL "$source_dataset_dir"/. "$constant_dataset_dir"/

python robo_manip_baselines/policy/wrench_predictor4_online/AddConstantPbToDataset.py \
  "$constant_dataset_dir" \
  --checkpoint "$wp4_checkpoint" \
  --overwrite

mkdir -p "$online_dataset_dir"
cp -aL "$source_dataset_dir"/. "$online_dataset_dir"/

python robo_manip_baselines/policy/wrench_predictor4_online/AddOnlinePbToDataset.py \
  "$online_dataset_dir" \
  0 \
  --checkpoint "$wp4_checkpoint" \
  --update_type gaussian_belief \
  --initial_std 0.25 \
  --num_points 16 \
  --beta 10 \
  --wrench_loss_weight 0.0 \
  --overwrite

echo "constant PB dataset: $constant_dataset_dir"
echo "Gaussian Online PB dataset: $online_dataset_dir"
