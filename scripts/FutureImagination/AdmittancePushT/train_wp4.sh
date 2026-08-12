#!/usr/bin/env bash
set -e

# Run from the repository root.

python robo_manip_baselines/bin/Train.py WrenchPredictor4 \
  --dataset_dir robo_manip_baselines/dataset/DatasetMujocoXarm7AdmittancePusht/training \
  --val_dataset_dir robo_manip_baselines/dataset/DatasetMujocoXarm7AdmittancePushtValidation/validation \
  --checkpoint_dir robo_manip_baselines/checkpoint/WrenchPredictor4/AdmittancePushT \
  --camera_names \
  --state_keys measured_eef_pose \
  --action_keys command_eef_pose \
  --image_feature_key measured_tblock_pose \
  --wrench_source_key measured_eef_wrench \
  --pb_dim 2 \
  --wrench_loss_weight 0.1 \
  --num_epochs 500 \
  --seed 42

source_dataset=robo_manip_baselines/dataset/DatasetMujocoXarm7AdmittancePusht
online_dataset=robo_manip_baselines/dataset/DatasetMujocoXarm7AdmittancePusht_online_pb2
constant_dataset=robo_manip_baselines/dataset/DatasetMujocoXarm7AdmittancePusht_constant_pb2
wp4_checkpoint=robo_manip_baselines/checkpoint/WrenchPredictor4/AdmittancePushT/policy_best.ckpt

cp -r "$source_dataset/." "$online_dataset"
cp -r "$source_dataset/." "$constant_dataset"

python robo_manip_baselines/policy/wrench_predictor4_online/AddOnlinePbToDataset.py \
  "$online_dataset" \
  0 \
  --checkpoint "$wp4_checkpoint" \
  --lr 2.5e-2 \
  --wrench_loss_weight 0.01 \
  --overwrite

python robo_manip_baselines/policy/wrench_predictor4_online/AddConstantPbToDataset.py \
  "$constant_dataset" \
  --checkpoint "$wp4_checkpoint" \
  --overwrite
