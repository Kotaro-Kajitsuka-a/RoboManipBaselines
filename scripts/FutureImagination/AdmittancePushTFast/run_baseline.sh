#!/usr/bin/env bash
set -e

# Train and evaluate the ordinary Diffusion Policy baseline on the Fast dataset.
base_dir=scripts/FutureImagination/AdmittancePushTFast

bash "$base_dir/train_baseline_seed42_52_62.sh"
bash "$base_dir/rollout_baseline_seed42_52_62.sh"
