#!/usr/bin/env bash
set -e

# Train the reproducible BetaVAE 9D with beta=2 and joint-position WP4.
train_dataset_dir=robo_manip_baselines/dataset/DatasetPushingPlaying/training
validation_dataset_dir=robo_manip_baselines/dataset/DatasetPushingPlaying/validation
vae_output_dir=robo_manip_baselines/checkpoint/ImageVAE/DatasetPushingPlaying_front_9_beta2
wp4_checkpoint_dir=robo_manip_baselines/checkpoint/WrenchPredictor4/DatasetPushingPlaying_front_image_vae_9_beta2_joint_pos

python robo_manip_baselines/misc/futureimagination/TrainImageVAE.py \
  --train_dataset_dir "$train_dataset_dir" \
  --validation_dataset_dir "$validation_dataset_dir" \
  --output_dir "$vae_output_dir" \
  --camera_name front \
  --latent_dim 16 \
  --beta 1.0

python robo_manip_baselines/misc/futureimagination/AddImageVAEFeature.py \
  "$train_dataset_dir" \
  --checkpoint "$vae_output_dir/final_model" \
  --camera_name front \
  --overwrite

python robo_manip_baselines/misc/futureimagination/AddImageVAEFeature.py \
  "$validation_dataset_dir" \
  --checkpoint "$vae_output_dir/final_model" \
  --camera_name front \
  --overwrite


python robo_manip_baselines/misc/futureimagination/MakeReconstructionVideos.py \
  "$train_dataset_dir" \
  "$vae_output_dir/final_model" \
  --camera_name front

python robo_manip_baselines/misc/futureimagination/MakeReconstructionVideos.py \
  "$validation_dataset_dir" \
  "$vae_output_dir/final_model" \
  --camera_name front
