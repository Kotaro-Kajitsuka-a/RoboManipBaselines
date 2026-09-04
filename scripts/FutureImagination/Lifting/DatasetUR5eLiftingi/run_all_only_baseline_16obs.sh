#!/usr/bin/env bash
set -e

# Run the 16-step-observation ordinary Diffusion Policy baseline only.
bash scripts/FutureImagination/Lifting/DatasetUR5eLiftingi/train_baseline_16obs_seed42_52_62.sh
bash scripts/FutureImagination/Lifting/DatasetUR5eLiftingi/rollout_baseline_16obs_seed42_52_62.sh
