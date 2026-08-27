#!/usr/bin/env bash
set -e

# Train only the current best Sorting VAE. This script does not train WP4.
train_dataset_dir=robo_manip_baselines/dataset/0826_DatasetSorting/training
validation_dataset_dir=robo_manip_baselines/dataset/0826_DatasetSorting/validation
vae_output_dir=robo_manip_baselines/checkpoint/ImageVAE/0826_DatasetSorting_front_16_plain_3stage_large

python robo_manip_baselines/misc/futureimagination/TrainImageVAE.py \
  --train_dataset_dir "$train_dataset_dir" \
  --validation_dataset_dir "$validation_dataset_dir" \
  --output_dir "$vae_output_dir" \
  --camera_name front \
  --latent_dim 16 \
  --architecture large \
  --num_epochs 80
