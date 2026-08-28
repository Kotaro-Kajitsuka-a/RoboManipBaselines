#!/usr/bin/env bash
set -e

# Add constant and causal online PBs to copies of the Pouch dataset.
source_training_dataset=robo_manip_baselines/dataset/0827_DatasetPouch/training
source_validation_dataset=robo_manip_baselines/dataset/0827_DatasetPouch/validation
online_training_dataset=robo_manip_baselines/dataset/0827_DatasetPouch_RightVAE9_Wp4JointPos_AdamOnlinePB/training
online_validation_dataset=robo_manip_baselines/dataset/0827_DatasetPouch_RightVAE9_Wp4JointPos_AdamOnlinePB/validation
constant_training_dataset=robo_manip_baselines/dataset/0827_DatasetPouch_RightVAE9_Wp4JointPos_ConstantPB/training
wp4_checkpoint=robo_manip_baselines/checkpoint/WrenchPredictor4/0827_DatasetPouch_right_image_vae_9_joint_pos/policy_best.ckpt

mkdir -p \
  "$online_training_dataset" \
  "$online_validation_dataset" \
  "$constant_training_dataset"
cp -aL "$source_training_dataset/." "$online_training_dataset/"
cp -aL "$source_validation_dataset/." "$online_validation_dataset/"
cp -aL "$source_training_dataset/." "$constant_training_dataset/"

python robo_manip_baselines/policy/wrench_predictor4_online/AddOnlinePbToDataset.py \
  "$online_training_dataset" \
  1 \
  --checkpoint "$wp4_checkpoint" \
  --lr 8e-3 \
  --wrench_loss_weight 0.0 \
  --overwrite

python robo_manip_baselines/policy/wrench_predictor4_online/AddConstantPbToDataset.py \
  "$constant_training_dataset" \
  --checkpoint "$wp4_checkpoint" \
  --overwrite

python robo_manip_baselines/policy/wrench_predictor4_online/AddOnlinePbToDataset.py \
  "$online_validation_dataset" \
  1 \
  --checkpoint "$wp4_checkpoint" \
  --lr 8e-3 \
  --wrench_loss_weight 0.0 \
  --overwrite

python robo_manip_baselines/misc/futureimagination/PlotOnlinePbDataset.py \
  "$online_training_dataset/WrenchPredObject0"

python robo_manip_baselines/misc/futureimagination/PlotOnlinePbDataset.py \
  "$online_training_dataset/WrenchPredObject1"

python robo_manip_baselines/misc/futureimagination/PlotOnlinePbDataset.py \
  "$online_training_dataset/WrenchPredObject2"

python robo_manip_baselines/misc/futureimagination/PlotOnlinePbDataset.py \
  "$online_validation_dataset/WrenchPredObject0"

python robo_manip_baselines/misc/futureimagination/PlotOnlinePbDataset.py \
  "$online_validation_dataset/WrenchPredObject1"

python robo_manip_baselines/misc/futureimagination/PlotOnlinePbDataset.py \
  "$online_validation_dataset/WrenchPredObject2"
