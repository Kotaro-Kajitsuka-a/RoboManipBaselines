#!/usr/bin/env bash
set -e

# Add constant and causal online PBs to copies of the PushingLine dataset.
source_training_dataset=robo_manip_baselines/dataset/DatasetPushingLine/training
source_validation_dataset=robo_manip_baselines/dataset/DatasetPushingLine/validation
online_training_dataset=robo_manip_baselines/dataset/DatasetPushingLine_FrontVAE9_Wp4JointPos_AdamOnlinePB/training
online_validation_dataset=robo_manip_baselines/dataset/DatasetPushingLine_FrontVAE9_Wp4JointPos_AdamOnlinePB/validation
constant_training_dataset=robo_manip_baselines/dataset/DatasetPushingLine_FrontVAE9_Wp4JointPos_ConstantPB/training
wp4_checkpoint=robo_manip_baselines/checkpoint/WrenchPredictor4/DatasetPushingLine_front_image_vae_9_joint_pos/policy_best.ckpt

mkdir -p \
  "$online_training_dataset" \
  "$online_validation_dataset" \
  "$constant_training_dataset"
cp -aL "$source_training_dataset/." "$online_training_dataset/"
cp -aL "$source_validation_dataset/." "$online_validation_dataset/"
cp -aL "$source_training_dataset/." "$constant_training_dataset/"

python robo_manip_baselines/policy/wrench_predictor4_online/AddOnlinePbToDataset.py \
  "$online_training_dataset" \
  2 \
  --checkpoint "$wp4_checkpoint" \
  --lr 8e-3 \
  --wrench_loss_weight 1.0 \
  --overwrite

python robo_manip_baselines/policy/wrench_predictor4_online/AddConstantPbToDataset.py \
  "$constant_training_dataset" \
  --checkpoint "$wp4_checkpoint" \
  --overwrite

python robo_manip_baselines/policy/wrench_predictor4_online/AddOnlinePbToDataset.py \
  "$online_validation_dataset" \
  2 \
  --checkpoint "$wp4_checkpoint" \
  --lr 8e-3 \
  --wrench_loss_weight 1.0 \
  --overwrite

python robo_manip_baselines/misc/futureimagination/PlotOnlinePbDataset.py \
  "$online_training_dataset/WrenchPredObject0" \
  --reference_object_ids 0 1 2 3 4

python robo_manip_baselines/misc/futureimagination/PlotOnlinePbDataset.py \
  "$online_training_dataset/WrenchPredObject1" \
  --reference_object_ids 0 1 2 3 4

python robo_manip_baselines/misc/futureimagination/PlotOnlinePbDataset.py \
  "$online_training_dataset/WrenchPredObject2" \
  --reference_object_ids 0 1 2 3 4

python robo_manip_baselines/misc/futureimagination/PlotOnlinePbDataset.py \
  "$online_training_dataset/WrenchPredObject3" \
  --reference_object_ids 0 1 2 3 4

python robo_manip_baselines/misc/futureimagination/PlotOnlinePbDataset.py \
  "$online_training_dataset/WrenchPredObject4" \
  --reference_object_ids 0 1 2 3 4

python robo_manip_baselines/misc/futureimagination/PlotOnlinePbDataset.py \
  "$online_validation_dataset/WrenchPredObject0" \
  --reference_object_ids 0 1 2 3 4

python robo_manip_baselines/misc/futureimagination/PlotOnlinePbDataset.py \
  "$online_validation_dataset/WrenchPredObject1" \
  --reference_object_ids 0 1 2 3 4

python robo_manip_baselines/misc/futureimagination/PlotOnlinePbDataset.py \
  "$online_validation_dataset/WrenchPredObject2" \
  --reference_object_ids 0 1 2 3 4

python robo_manip_baselines/misc/futureimagination/PlotOnlinePbDataset.py \
  "$online_validation_dataset/WrenchPredObject3" \
  --reference_object_ids 0 1 2 3 4

python robo_manip_baselines/misc/futureimagination/PlotOnlinePbDataset.py \
  "$online_validation_dataset/WrenchPredObject4" \
  --reference_object_ids 0 1 2 3 4
