#!/usr/bin/env bash
set -e

# Run from the repository root after train_seed42_52_62.sh.
# Evaluate ordinary Diffusion Policy without online PB adaptation.
eval_timestamp=$(date +%Y%m%d_%H%M%S)
eval_dir="robo_manip_baselines/dataset/tests/FutureImagination/AdmittancePushTOracle/DP_baseline_eval_${eval_timestamp}"
rmb_dir="$eval_dir/rmb"
mkdir -p "$rmb_dir"
rollout_start_marker="$eval_dir/.rollout_start"
touch "$rollout_start_marker"

# Run 10 episodes for T0-T3 under every training seed.
for train_seed in 42 52 62; do

  python robo_manip_baselines/bin/Rollout.py \
    DiffusionPolicy MujocoXarm7AdmittancePusht_T0 \
    --demo_name "AdmittancePushTOracleBaseline_trainseed${train_seed}_rollseed42_T0" \
    --checkpoint "robo_manip_baselines/checkpoint/DiffusionPolicy/AdmittancePushTOracle/baseline_seed${train_seed}/policy_last.ckpt" \
    --world_idx_list {70..79} \
    --seed 42 \
    --auto_exit \
    --max_duration 22 \
    --result_filename "$eval_dir/baseline_trainseed${train_seed}_rollseed42_T0.yaml" \
    --save_rollout \
    --no_plot \
    --no_render

  python robo_manip_baselines/bin/Rollout.py \
    DiffusionPolicy MujocoXarm7AdmittancePusht_T1 \
    --demo_name "AdmittancePushTOracleBaseline_trainseed${train_seed}_rollseed42_T1" \
    --checkpoint "robo_manip_baselines/checkpoint/DiffusionPolicy/AdmittancePushTOracle/baseline_seed${train_seed}/policy_last.ckpt" \
    --world_idx_list {170..179} \
    --seed 42 \
    --auto_exit \
    --max_duration 22 \
    --result_filename "$eval_dir/baseline_trainseed${train_seed}_rollseed42_T1.yaml" \
    --save_rollout \
    --no_plot \
    --no_render

  python robo_manip_baselines/bin/Rollout.py \
    DiffusionPolicy MujocoXarm7AdmittancePusht_T2 \
    --demo_name "AdmittancePushTOracleBaseline_trainseed${train_seed}_rollseed42_T2" \
    --checkpoint "robo_manip_baselines/checkpoint/DiffusionPolicy/AdmittancePushTOracle/baseline_seed${train_seed}/policy_last.ckpt" \
    --world_idx_list {270..279} \
    --seed 42 \
    --auto_exit \
    --max_duration 22 \
    --result_filename "$eval_dir/baseline_trainseed${train_seed}_rollseed42_T2.yaml" \
    --save_rollout \
    --no_plot \
    --no_render

  python robo_manip_baselines/bin/Rollout.py \
    DiffusionPolicy MujocoXarm7AdmittancePusht_T3 \
    --demo_name "AdmittancePushTOracleBaseline_trainseed${train_seed}_rollseed42_T3" \
    --checkpoint "robo_manip_baselines/checkpoint/DiffusionPolicy/AdmittancePushTOracle/baseline_seed${train_seed}/policy_last.ckpt" \
    --world_idx_list {370..379} \
    --seed 42 \
    --auto_exit \
    --max_duration 22 \
    --result_filename "$eval_dir/baseline_trainseed${train_seed}_rollseed42_T3.yaml" \
    --save_rollout \
    --no_plot \
    --no_render

  for object_id in 0 1 2 3; do
    object_dir="$rmb_dir/WrenchPredObject${object_id}"
    mkdir -p "$object_dir"
    find robo_manip_baselines/dataset \
      -mindepth 1 \
      -maxdepth 1 \
      -type d \
      -name "RolloutDiffusionPolicy_AdmittancePushTOracleBaseline_trainseed${train_seed}_rollseed42_T${object_id}_20*" \
      -newer "$rollout_start_marker" \
      -exec mv -- {} "$object_dir" \;
  done
done
