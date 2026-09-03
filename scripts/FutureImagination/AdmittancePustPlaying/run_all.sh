#!/usr/bin/env bash
set -e

# Run the complete state-based AdmittancePustPlaying online-PB experiment.
base_dir=scripts/FutureImagination/AdmittancePustPlaying

bash "$base_dir/train_wp4.sh"
bash "$base_dir/prepare_dataset.sh"
bash "$base_dir/train_DP_online_pb_seed42_52_62.sh"
bash "$base_dir/rollout_DP_online_pb_seed42_52_62.sh"
