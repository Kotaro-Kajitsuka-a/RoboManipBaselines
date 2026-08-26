#!/usr/bin/env bash
set -e

# Check the existing legacy-dataset and VAE assets reused by this experiment.
source_dataset=robo_manip_baselines/dataset/LiftingAB_B_only
validation_dataset=robo_manip_baselines/dataset/LiftingAB_B_only_Validation
vae_checkpoint=robo_manip_baselines/checkpoint/ImageVAE/LiftingAB_B_only_9/final_model

if [ ! -d "$source_dataset" ]; then
  echo "Missing dataset: $source_dataset"
  exit 1
fi

if [ ! -d "$validation_dataset" ]; then
  echo "Missing validation dataset: $validation_dataset"
  exit 1
fi

if [ ! -d "$vae_checkpoint" ]; then
  echo "Missing VAE checkpoint: $vae_checkpoint"
  exit 1
fi

echo "All required dataset and VAE files exist."
