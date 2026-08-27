#!/usr/bin/env bash
set -e

# Retrain only the Online-PB Diffusion Policy. Reuse the existing WP4 and
# Constant-PB DP checkpoints, then roll out Online and Constant training.
bash scripts/FutureImagination/LiftingDpImage/DatasetUR5eLiftingiSGD/train_DP_online_pb_seed42_52_62.sh
bash scripts/FutureImagination/LiftingDpImage/DatasetUR5eLiftingiSGD/rollout_DP_online_pb_seed42_52_62.sh
bash scripts/FutureImagination/LiftingDpImage/DatasetUR5eLiftingiSGD/rollout_constant_pb_seed42_52_62.sh
