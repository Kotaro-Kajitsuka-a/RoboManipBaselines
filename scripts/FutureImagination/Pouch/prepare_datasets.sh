#!/usr/bin/env bash
set -e

# Recreate the adopted beta=2 VAE feature, then add beta=2 WP4 PBs to copies.
source_training_dataset=robo_manip_baselines/dataset/0827_DatasetPouch/training
source_validation_dataset=robo_manip_baselines/dataset/0827_DatasetPouch/validation
online_training_dataset=robo_manip_baselines/dataset/0827_DatasetPouch_RightVAE9Beta2_Wp4JointPos_AdamOnlinePB/training
online_validation_dataset=robo_manip_baselines/dataset/0827_DatasetPouch_RightVAE9Beta2_Wp4JointPos_AdamOnlinePB/validation
constant_training_dataset=robo_manip_baselines/dataset/0827_DatasetPouch_RightVAE9Beta2_Wp4JointPos_ConstantPB/training
image_vae_checkpoint=robo_manip_baselines/checkpoint/ImageVAE/0827_DatasetPouch_right_9_beta2/final_model
wp4_checkpoint=robo_manip_baselines/checkpoint/WrenchPredictor4/0827_DatasetPouch_right_image_vae_9_beta2_joint_pos/policy_best.ckpt

python robo_manip_baselines/misc/futureimagination/AddImageVAEFeature.py \
  "$source_training_dataset" \
  --checkpoint "$image_vae_checkpoint" \
  --camera_name right \
  --overwrite

python robo_manip_baselines/misc/futureimagination/AddImageVAEFeature.py \
  "$source_validation_dataset" \
  --checkpoint "$image_vae_checkpoint" \
  --camera_name right \
  --overwrite

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
  "$online_training_dataset/WrenchPredObject0" \
  --reference_object_ids 0 1 2

python robo_manip_baselines/misc/futureimagination/PlotOnlinePbDataset.py \
  "$online_training_dataset/WrenchPredObject1" \
  --reference_object_ids 0 1 2

python robo_manip_baselines/misc/futureimagination/PlotOnlinePbDataset.py \
  "$online_training_dataset/WrenchPredObject2" \
  --reference_object_ids 0 1 2

python robo_manip_baselines/misc/futureimagination/PlotOnlinePbDataset.py \
  "$online_validation_dataset/WrenchPredObject0" \
  --reference_object_ids 0 1 2

python robo_manip_baselines/misc/futureimagination/PlotOnlinePbDataset.py \
  "$online_validation_dataset/WrenchPredObject1" \
  --reference_object_ids 0 1 2

python robo_manip_baselines/misc/futureimagination/PlotOnlinePbDataset.py \
  "$online_validation_dataset/WrenchPredObject2" \
  --reference_object_ids 0 1 2
