#!/usr/bin/env bash
set -e

# Run the complete AdmittancePushT Fast experiment except for the ordinary
# Diffusion Policy baseline.
base_dir=scripts/FutureImagination/AdmittancePushTFast

bash "$base_dir/train_wp4.sh"
bash "$base_dir/train_DP_online_pb_seed42_52_62.sh"
bash "$base_dir/train_constant_pb_seed42_52_62.sh"
bash "$base_dir/rollout_DP_online_pb_seed42_52_62.sh"
bash "$base_dir/rollout_constant_pb_seed42_52_62.sh"
bash "$base_dir/rollout_DP_oracle_pb_seed42_52_62.sh"
