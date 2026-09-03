#!/usr/bin/env bash
set -e

# Run only the complete fast-constant/slow-online proposed method.
base_dir=scripts/FutureImagination/AdmittancePushtAB

bash "$base_dir/train_wp4.sh"
bash "$base_dir/prepare_dataset_fast_slow.sh"
bash "$base_dir/train_fast_slow_seed42_52_62.sh"
bash "$base_dir/rollout_fast_slow_seed42_52_62.sh"
