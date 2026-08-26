#!/usr/bin/env bash
set -e

# Check the legacy training and validation datasets used by this experiment.
source_dataset=robo_manip_baselines/dataset/LiftingAB_B_only
validation_dataset=robo_manip_baselines/dataset/LiftingAB_B_only_Validation

if [ ! -d "$source_dataset" ]; then
  echo "Missing dataset: $source_dataset"
  exit 1
fi

if [ ! -d "$validation_dataset" ]; then
  echo "Missing validation dataset: $validation_dataset"
  exit 1
fi

echo "All required datasets exist."
