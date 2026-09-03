#!/usr/bin/env bash
set -e

# Run the complete state-based AdmittancePushtAB experiment.
base_dir=scripts/FutureImagination/AdmittancePushtAB

bash "$base_dir/train_wp4.sh"
bash "$base_dir/prepare_datasets.sh"
bash "$base_dir/train_baseline_seed42_52_62.sh"
bash "$base_dir/train_DP_online_pb_seed42_52_62.sh"
bash "$base_dir/train_constant_pb_seed42_52_62.sh"
bash "$base_dir/rollout_baseline_seed42_52_62.sh"
bash "$base_dir/rollout_DP_online_pb_seed42_52_62.sh"
bash "$base_dir/rollout_constant_pb_seed42_52_62.sh"
