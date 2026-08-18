#!/usr/bin/env bash
set -e

# Run from the repository root with the virtual environment activated.
python robo_manip_baselines/misc/futureimagination/TrainSD3LatentAE.py \
  --train_dataset_dir robo_manip_baselines/dataset/LiftingAB_B_only \
  --validation_dataset_dir robo_manip_baselines/dataset/LiftingAB_B_only_Validation \
  --output_dir robo_manip_baselines/checkpoint/SD3LatentAE/LiftingAB_B_only_hand_12 \
  --image_feature_key sd3_vae_hand \
  --latent_dim 12
