#!/usr/bin/env bash
set -e

# Recompute the PB labels from the new dataset's training split. Do not reuse
# a previously PB-labeled dataset from another Diffusion Policy experiment.
online_pb_lr=${ONLINE_PB_LR:-6e-3}
experiment_suffix=${EXPERIMENT_SUFFIX:-}
source_dataset=robo_manip_baselines/dataset/DatasetMujocoUR5eLiftingi/training
online_dataset="robo_manip_baselines/dataset/DatasetMujocoUR5eLiftingi_LeftScratchVAE9_Wp4JointPos_DpEefPose_AdamOnlinePB${experiment_suffix}"
constant_dataset="robo_manip_baselines/dataset/DatasetMujocoUR5eLiftingi_LeftScratchVAE9_Wp4JointPos_DpEefPose_ConstantPB${experiment_suffix}"
wp4_checkpoint=robo_manip_baselines/checkpoint/WrenchPredictor4/DatasetMujocoUR5eLiftingi_left_image_vae_9_joint_pos/policy_best.ckpt

test -d "$source_dataset"
test -f "$wp4_checkpoint"
test ! -e "$online_dataset"
test ! -e "$constant_dataset"

mkdir -p "$online_dataset" "$constant_dataset"
cp -aL "$source_dataset/." "$online_dataset/"
cp -aL "$source_dataset/." "$constant_dataset/"

python scripts/FutureImagination/LiftingImage/DatasetUR5eLiftingiLeftScratchVAE9Wp4JointPosDpEefPose/add_online_pb_to_dataset.py \
  "$online_dataset" \
  0 \
  --checkpoint "$wp4_checkpoint" \
  --lr "$online_pb_lr" \
  --wrench_loss_weight 0.0 \
  --overwrite

python robo_manip_baselines/policy/wrench_predictor4_online/AddConstantPbToDataset.py \
  "$constant_dataset" \
  --checkpoint "$wp4_checkpoint" \
  --overwrite
