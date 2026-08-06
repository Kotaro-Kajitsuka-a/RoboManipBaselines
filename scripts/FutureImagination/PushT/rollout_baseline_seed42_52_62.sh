#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

checkpoint_root=robo_manip_baselines/checkpoint/DiffusionPolicy
eval_dir=robo_manip_baselines/dataset/tests/FutureImagination/DatasetMujocoXarm7Pusht_eval_rollseed42_52_62

mkdir -p "$eval_dir"

for seed in 42 52 62; do
  checkpoint="$checkpoint_root/DatasetMujocoXarm7Pusht_baseline_e500_seed${seed}_20260806/policy_last.ckpt"

  python robo_manip_baselines/bin/Rollout.py DiffusionPolicy MujocoXarm7Pusht_T0 \
    --checkpoint "$checkpoint" \
    --world_idx_list {70..79} \
    --seed 42 \
    --auto_exit \
    --max_duration 22 \
    --save_rollout \
    --result_filename "$eval_dir/baseline_trainseed${seed}_rollseed42_T0.yaml" \
    --no_plot \
    --no_render

  python robo_manip_baselines/bin/Rollout.py DiffusionPolicy MujocoXarm7Pusht_T1 \
    --checkpoint "$checkpoint" \
    --world_idx_list {170..179} \
    --seed 42 \
    --auto_exit \
    --max_duration 22 \
    --save_rollout \
    --result_filename "$eval_dir/baseline_trainseed${seed}_rollseed42_T1.yaml" \
    --no_plot \
    --no_render

  python robo_manip_baselines/bin/Rollout.py DiffusionPolicy MujocoXarm7Pusht_T2 \
    --checkpoint "$checkpoint" \
    --world_idx_list {270..279} \
    --seed 42 \
    --auto_exit \
    --max_duration 22 \
    --save_rollout \
    --result_filename "$eval_dir/baseline_trainseed${seed}_rollseed42_T2.yaml" \
    --no_plot \
    --no_render

  python robo_manip_baselines/bin/Rollout.py DiffusionPolicy MujocoXarm7Pusht_T3 \
    --checkpoint "$checkpoint" \
    --world_idx_list {370..379} \
    --seed 42 \
    --auto_exit \
    --max_duration 22 \
    --save_rollout \
    --result_filename "$eval_dir/baseline_trainseed${seed}_rollseed42_T3.yaml" \
    --no_plot \
    --no_render

  python robo_manip_baselines/bin/Rollout.py DiffusionPolicy MujocoXarm7Pusht_T4 \
    --checkpoint "$checkpoint" \
    --world_idx_list {470..479} \
    --seed 42 \
    --auto_exit \
    --max_duration 22 \
    --save_rollout \
    --result_filename "$eval_dir/baseline_trainseed${seed}_rollseed42_T4.yaml" \
    --no_plot \
    --no_render
done
