#!/usr/bin/env bash
set -e

# Run the complete Gaussian EEF-pose study except for the existing baseline.
bash scripts/FutureImagination/Lifting/DatasetUR5eLiftingiGausian/train_wp4.sh
bash scripts/FutureImagination/Lifting/DatasetUR5eLiftingiGausian/prepare_dataset.sh
bash scripts/FutureImagination/Lifting/DatasetUR5eLiftingiGausian/train_DP_online_pb_seed42_52_62.sh
bash scripts/FutureImagination/Lifting/DatasetUR5eLiftingiGausian/train_constant_pb_seed42_52_62.sh
bash scripts/FutureImagination/Lifting/DatasetUR5eLiftingiGausian/rollout_DP_online_pb_seed42_52_62.sh
bash scripts/FutureImagination/Lifting/DatasetUR5eLiftingiGausian/rollout_constant_pb_seed42_52_62.sh
