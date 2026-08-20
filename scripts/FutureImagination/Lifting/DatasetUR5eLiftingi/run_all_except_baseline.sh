#!/usr/bin/env bash
set -e

# Run the complete joint-pos state/PB experiment except for the ordinary
# baseline, which is shared with the left-image experiment.
bash scripts/FutureImagination/Lifting/DatasetUR5eLiftingi/train_wp4.sh
bash scripts/FutureImagination/Lifting/DatasetUR5eLiftingi/train_DP_online_pb_seed42_52_62.sh
bash scripts/FutureImagination/Lifting/DatasetUR5eLiftingi/train_constant_pb_seed42_52_62.sh
bash scripts/FutureImagination/Lifting/DatasetUR5eLiftingi/rollout_DP_online_pb_seed42_52_62.sh
bash scripts/FutureImagination/Lifting/DatasetUR5eLiftingi/rollout_constant_pb_seed42_52_62.sh
