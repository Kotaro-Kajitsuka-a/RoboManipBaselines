#!/usr/bin/env bash
set -e

# Train the hand-camera ImageVAE 9D and WP4 on the legacy dataset.
train_dataset_dir=robo_manip_baselines/dataset/LiftingAB_B_only
validation_dataset_dir=robo_manip_baselines/dataset/LiftingAB_B_only_Validation
vae_output_dir=robo_manip_baselines/checkpoint/ImageVAE/LiftingAB_B_only_hand_9
wp4_checkpoint_dir=robo_manip_baselines/checkpoint/WrenchPredictor4/LiftingAB_B_only_hand_image_vae_9

python robo_manip_baselines/misc/futureimagination/TrainImageVAE.py \
  --train_dataset_dir "$train_dataset_dir" \
  --validation_dataset_dir "$validation_dataset_dir" \
  --output_dir "$vae_output_dir" \
  --camera_name hand \
  --latent_dim 9

python robo_manip_baselines/misc/futureimagination/AddImageVAEFeature.py \
  "$train_dataset_dir" \
  --checkpoint "$vae_output_dir/final_model" \
  --camera_name hand \
  --overwrite

python robo_manip_baselines/misc/futureimagination/AddImageVAEFeature.py \
  "$validation_dataset_dir" \
  --checkpoint "$vae_output_dir/final_model" \
  --camera_name hand \
  --overwrite

python robo_manip_baselines/bin/Train.py WrenchPredictor4 \
  --dataset_dir "$train_dataset_dir" \
  --val_dataset_dir "$validation_dataset_dir" \
  --checkpoint_dir "$wp4_checkpoint_dir" \
  --camera_names \
  --state_keys measured_eef_pose measured_gripper_joint_pos \
  --action_keys command_eef_pose command_gripper_joint_pos \
  --image_feature_key image_vae_hand_9 \
  --wrench_source_key measured_eef_wrench \
  --pb_dim 1 \
  --wrench_loss_weight 0.1 \
  --num_epochs 500
