#!/usr/bin/env bash
set -e

# Run the complete image-based DP experiment except for the ordinary baseline.
# Reuse the existing EEF-pose WP4 policy_best.ckpt.
bash scripts/FutureImagination/LiftingDpImage/DatasetUR5eLiftingi/prepare_datasets.sh
bash scripts/FutureImagination/LiftingDpImage/DatasetUR5eLiftingi/train_DP_online_pb_seed42_52_62.sh
bash scripts/FutureImagination/LiftingDpImage/DatasetUR5eLiftingi/train_constant_pb_seed42_52_62.sh
bash scripts/FutureImagination/LiftingDpImage/DatasetUR5eLiftingi/rollout_DP_online_pb_seed42_52_62.sh
bash scripts/FutureImagination/LiftingDpImage/DatasetUR5eLiftingi/rollout_constant_pb_seed42_52_62.sh
