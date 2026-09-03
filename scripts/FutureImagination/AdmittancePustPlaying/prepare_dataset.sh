#!/usr/bin/env bash
set -e

# Run from the repository root after train_wp4.sh. Keep the source DP dataset
# untouched and add Playing-WP4 online PB trajectories to an independent copy.
source_dataset=robo_manip_baselines/dataset/DatasetMujocoXarm7AdmittancePusht
online_dataset=robo_manip_baselines/dataset/DatasetMujocoXarm7AdmittancePusht_Playing_OnlinePB
wp4_checkpoint=robo_manip_baselines/checkpoint/WrenchPredictor4/AdmittancePustPlaying/policy_best.ckpt

mkdir -p "$online_dataset"
cp -aL "$source_dataset/." "$online_dataset/"

python robo_manip_baselines/policy/wrench_predictor4_online/AddOnlinePbToDataset.py \
  "$online_dataset" \
  0 \
  --checkpoint "$wp4_checkpoint" \
  --lr 2.5e-2 \
  --wrench_loss_weight 1.0 \
  --overwrite
