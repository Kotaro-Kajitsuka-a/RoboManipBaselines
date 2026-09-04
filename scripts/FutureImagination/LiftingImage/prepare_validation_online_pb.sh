#!/usr/bin/env bash
set -e

# Run from the repository root with the virtual environment activated.
source_dataset=robo_manip_baselines/dataset/LiftingAB_B_only_Validation
online_dataset=robo_manip_baselines/dataset/LiftingImageAB_B_only_Validation_OnlinePB
wp4_checkpoint=robo_manip_baselines/checkpoint/WrenchPredictor4/LiftingAB_B_only_hand_image_vae_9/policy_best.ckpt

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
  "$online_dataset/WrenchPredObject0" \
  --reference_object_ids 0 1 2

python robo_manip_baselines/misc/futureimagination/PlotOnlinePbDataset.py \
  "$online_dataset/WrenchPredObject1" \
  --reference_object_ids 0 1 2

python robo_manip_baselines/misc/futureimagination/PlotOnlinePbDataset.py \
  "$online_dataset/WrenchPredObject2" \
  --reference_object_ids 0 1 2
