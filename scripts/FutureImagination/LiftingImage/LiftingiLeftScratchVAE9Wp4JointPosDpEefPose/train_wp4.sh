#!/usr/bin/env bash
set -e

# Retrain WP4 with the existing left-camera VAE 9D feature on the legacy
# lifting dataset. State/action inputs are both joint positions.
train_dataset_dir=robo_manip_baselines/dataset/LiftingAB_B_only
validation_dataset_dir=robo_manip_baselines/dataset/LiftingAB_B_only_Validation
wp4_checkpoint_dir=robo_manip_baselines/checkpoint/WrenchPredictor4/LiftingAB_B_only_left_image_vae_9_joint_pos

python robo_manip_baselines/bin/Train.py WrenchPredictor4 \
  --dataset_dir "$train_dataset_dir" \
  --val_dataset_dir "$validation_dataset_dir" \
  --checkpoint_dir "$wp4_checkpoint_dir" \
  --camera_names \
  --state_keys measured_joint_pos \
  --action_keys command_joint_pos \
  --image_feature_key image_vae_9 \
  --wrench_source_key measured_eef_wrench \
  --pb_dim 1 \
  --wrench_loss_weight 0.1 \
  --num_epochs 500

python robo_manip_baselines/policy/wrench_predictor4/EvalWrenchPredictor4ImageFeatureSweepDir.py \
  "$wp4_checkpoint_dir/policy_best.ckpt" \
  "$validation_dataset_dir" \
  --material_object_ids 0 1 2
