#!/usr/bin/env bash
set -e

# Run from the repository root after train_wp4.sh. Keep the source dataset
# untouched and prepare independent copies for the online- and constant-PB DPs.
source_dataset=robo_manip_baselines/dataset/DatasetAdmittancePushtAB/training
online_dataset=robo_manip_baselines/dataset/DatasetAdmittancePushtAB_OnlinePB
constant_dataset=robo_manip_baselines/dataset/DatasetAdmittancePushtAB_ConstantPB
wp4_checkpoint=robo_manip_baselines/checkpoint/WrenchPredictor4/AdmittancePushtAB/policy_best.ckpt

mkdir -p "$online_dataset" "$constant_dataset"
cp -aL "$source_dataset/." "$online_dataset/"
cp -aL "$source_dataset/." "$constant_dataset/"

python robo_manip_baselines/policy/wrench_predictor4_online/AddOnlinePbToDataset.py \
  "$online_dataset" \
  0 \
  --checkpoint "$wp4_checkpoint" \
  --lr 2.5e-2 \
  --wrench_loss_weight 1.0 \
  --overwrite

python robo_manip_baselines/policy/wrench_predictor4_online/AddConstantPbToDataset.py \
  "$constant_dataset" \
  --checkpoint "$wp4_checkpoint" \
  --overwrite
