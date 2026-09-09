#!/usr/bin/env bash
set -e

# Run from the repository root after post_train_wp4.sh.
wp4_checkpoint=robo_manip_baselines/checkpoint/WrenchPredictor4/DatasetPushingLine_PostTrain_front_image_vae_9_joint_pos/policy_best.ckpt
source_training_dataset=robo_manip_baselines/dataset/DatasetPushingLine_PostTrain/training
source_validation_dataset=robo_manip_baselines/dataset/DatasetPushingLine_PostTrain/validation
online_training_dataset=robo_manip_baselines/dataset/DatasetPushingLine_PostTrain_FrontVAE9_Wp4JointPos_AdamOnlinePB/training
online_validation_dataset=robo_manip_baselines/dataset/DatasetPushingLine_PostTrain_FrontVAE9_Wp4JointPos_AdamOnlinePB/validation
constant_training_dataset=robo_manip_baselines/dataset/DatasetPushingLine_PostTrain_FrontVAE9_Wp4JointPos_ConstantPB/training

if [ ! -f "$wp4_checkpoint" ]; then
  echo "WP4 checkpoint not found: $wp4_checkpoint" >&2
  exit 1
fi

mkdir -p \
  "$online_training_dataset" \
  "$online_validation_dataset" \
  "$constant_training_dataset"
# Mirror the combined dataset, preserving its demo/rollout and seed directories.
rsync -aL --ignore-times --delete "$source_training_dataset/" "$online_training_dataset/"
rsync -aL --ignore-times --delete "$source_validation_dataset/" "$online_validation_dataset/"
rsync -aL --ignore-times --delete "$source_training_dataset/" "$constant_training_dataset/"

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

# Plot the five objects separately in each demo/rollout group.
for dataset_group in \
  "$online_training_dataset/demo" \
  "$online_training_dataset/rollout/seed42" \
  "$online_training_dataset/rollout/seed52" \
  "$online_training_dataset/rollout/seed62" \
  "$online_validation_dataset/demo" \
  "$online_validation_dataset/rollout/baseline/seed62"
do
  for object_id in 0 1 2 3 4
  do
    python robo_manip_baselines/misc/futureimagination/PlotOnlinePbDataset.py \
      "$dataset_group/WrenchPredObject$object_id" \
      --reference_object_ids 0 1 2 3 4
  done
done
