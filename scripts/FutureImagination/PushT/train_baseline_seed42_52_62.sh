#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

dataset_dir=robo_manip_baselines/dataset/DatasetMujocoXarm7Pusht
checkpoint_root=robo_manip_baselines/checkpoint/DiffusionPolicy

for seed in 42 52 62; do
  python robo_manip_baselines/bin/Train.py DiffusionPolicy \
    --dataset_dir "$dataset_dir" \
    --checkpoint_dir "$checkpoint_root/DatasetMujocoXarm7Pusht_baseline_e500_seed${seed}_20260806" \
    --camera_names \
    --state_keys measured_eef_pose measured_tblock_pose \
    --action_keys command_eef_pose \
    --skip 3 \
    --num_epochs 500 \
    --lr 1e-4 \
    --train_ratio 1.0 \
    --val_ratio 0.01 \
    --seed "$seed"
done
