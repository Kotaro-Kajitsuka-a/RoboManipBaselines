#!/usr/bin/env bash
set -e

# Run from the repository root with the virtual environment activated.
source_dataset=robo_manip_baselines/dataset/LiftingAB_B_only_Validation
online_dataset=robo_manip_baselines/dataset/LiftingAB_B_only_Validation_OnlinePB
wp4_checkpoint=robo_manip_baselines/checkpoint/WrenchPredictor4/LiftingAB_B_only/policy_best.ckpt

mkdir -p "$online_dataset"
cp -aL "$source_dataset/." "$online_dataset/"

python robo_manip_baselines/policy/wrench_predictor4_online/AddOnlinePbToDataset.py \
  "$online_dataset" \
  0 \
  --checkpoint "$wp4_checkpoint" \
  --lr 6e-3 \
  --overwrite

python robo_manip_baselines/misc/futureimagination/PlotOnlinePbDataset.py \
  "$online_dataset/WrenchPredObject0"

python robo_manip_baselines/misc/futureimagination/PlotOnlinePbDataset.py \
  "$online_dataset/WrenchPredObject1"

python robo_manip_baselines/misc/futureimagination/PlotOnlinePbDataset.py \
  "$online_dataset/WrenchPredObject2"
