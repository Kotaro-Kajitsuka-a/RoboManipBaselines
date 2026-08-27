#!/usr/bin/env bash
set -e

# Run from the repository root with the virtual environment activated.
train_dataset_dir=robo_manip_baselines/dataset/LiftingAB_B_only
validation_dataset_dir=robo_manip_baselines/dataset/LiftingAB_B_only_Validation
ae_checkpoint=robo_manip_baselines/checkpoint/SD3LatentAE/LiftingAB_B_only_left_12/final_model
wp4_checkpoint_dir=robo_manip_baselines/checkpoint/WrenchPredictor4/LiftingAB_B_only_left_sd3_ae_12

python robo_manip_baselines/misc/futureimagination/AddSD3LatentAEFeature.py \
  "$train_dataset_dir" \
  --checkpoint "$ae_checkpoint" \
  --source_key sd3_vae_left_128x96 \
  --output_key sd3_vae_left_ae_12 \
  --overwrite

python robo_manip_baselines/misc/futureimagination/AddSD3LatentAEFeature.py \
  "$validation_dataset_dir" \
  --checkpoint "$ae_checkpoint" \
  --source_key sd3_vae_left_128x96 \
  --output_key sd3_vae_left_ae_12 \
  --overwrite

python robo_manip_baselines/bin/Train.py WrenchPredictor4 \
  --dataset_dir "$train_dataset_dir" \
  --val_dataset_dir "$validation_dataset_dir" \
  --checkpoint_dir "$wp4_checkpoint_dir" \
  --camera_names \
  --state_keys measured_eef_pose measured_gripper_joint_pos \
  --action_keys command_eef_pose command_gripper_joint_pos \
  --image_feature_key sd3_vae_left_ae_12 \
  --wrench_source_key measured_eef_wrench \
  --pb_dim 1 \
  --wrench_loss_weight 0.1 \
  --skip_wrench_preprocessing \
  --num_epochs 500
