#!/usr/bin/env bash
set -e

# Run from the repository root after train_baseline_seed42_52_62.sh.
# Evaluate ordinary Diffusion Policy without online PB adaptation.
eval_timestamp=$(date +%Y%m%d_%H%M%S)
eval_dir="robo_manip_baselines/dataset/tests/FutureImagination/LiftingAB_B_only/DP_baseline_eval_${eval_timestamp}"
rmb_dir="$eval_dir/rmb"
mkdir -p "$rmb_dir"
rollout_start_marker="$eval_dir/.rollout_start"
touch "$rollout_start_marker"

# Run 10 episodes for each object under every training seed.
for train_seed in 42 52 62; do

  python robo_manip_baselines/bin/Rollout.py \
    DiffusionPolicy MujocoUR5eLiftingi_I0 \
    --demo_name "MujocoUR5eLiftingi_I0_baseline_trainseed${train_seed}" \
    --checkpoint "robo_manip_baselines/checkpoint/DiffusionPolicy/LiftingAB_B_only_Baseline_seed${train_seed}/policy_last.ckpt" \
    --seed 42 \
    --world_idx_list {70..79} \
    --auto_exit \
    --max_duration 10 \
    --save_rollout \
    --result_filename "$eval_dir/baseline_trainseed${train_seed}_I0.yaml" \
    --no_plot \
    --no_render

  python robo_manip_baselines/bin/Rollout.py \
    DiffusionPolicy MujocoUR5eLiftingi_I1 \
    --demo_name "MujocoUR5eLiftingi_I1_baseline_trainseed${train_seed}" \
    --checkpoint "robo_manip_baselines/checkpoint/DiffusionPolicy/LiftingAB_B_only_Baseline_seed${train_seed}/policy_last.ckpt" \
    --seed 42 \
    --world_idx_list {170..179} \
    --auto_exit \
    --max_duration 10 \
    --save_rollout \
    --result_filename "$eval_dir/baseline_trainseed${train_seed}_I1.yaml" \
    --no_plot \
    --no_render

  python robo_manip_baselines/bin/Rollout.py \
    DiffusionPolicy MujocoUR5eLiftingi_I2 \
    --demo_name "MujocoUR5eLiftingi_I2_baseline_trainseed${train_seed}" \
    --checkpoint "robo_manip_baselines/checkpoint/DiffusionPolicy/LiftingAB_B_only_Baseline_seed${train_seed}/policy_last.ckpt" \
    --seed 42 \
    --world_idx_list {270..279} \
    --auto_exit \
    --max_duration 10 \
    --save_rollout \
    --result_filename "$eval_dir/baseline_trainseed${train_seed}_I2.yaml" \
    --no_plot \
    --no_render

  python robo_manip_baselines/bin/Rollout.py \
    DiffusionPolicy MujocoUR5eLiftingi_I4 \
    --demo_name "MujocoUR5eLiftingi_I4_baseline_trainseed${train_seed}" \
    --checkpoint "robo_manip_baselines/checkpoint/DiffusionPolicy/LiftingAB_B_only_Baseline_seed${train_seed}/policy_last.ckpt" \
    --seed 42 \
    --world_idx_list {470..479} \
    --auto_exit \
    --max_duration 10 \
    --save_rollout \
    --result_filename "$eval_dir/baseline_trainseed${train_seed}_I4.yaml" \
    --no_plot \
    --no_render

  python robo_manip_baselines/bin/Rollout.py \
    DiffusionPolicy MujocoUR5eLiftingi_I5 \
    --demo_name "MujocoUR5eLiftingi_I5_baseline_trainseed${train_seed}" \
    --checkpoint "robo_manip_baselines/checkpoint/DiffusionPolicy/LiftingAB_B_only_Baseline_seed${train_seed}/policy_last.ckpt" \
    --seed 42 \
    --world_idx_list {570..579} \
    --auto_exit \
    --max_duration 10 \
    --save_rollout \
    --result_filename "$eval_dir/baseline_trainseed${train_seed}_I5.yaml" \
    --no_plot \
    --no_render

  python robo_manip_baselines/bin/Rollout.py \
    DiffusionPolicy MujocoUR5eLiftingi_I6 \
    --demo_name "MujocoUR5eLiftingi_I6_baseline_trainseed${train_seed}" \
    --checkpoint "robo_manip_baselines/checkpoint/DiffusionPolicy/LiftingAB_B_only_Baseline_seed${train_seed}/policy_last.ckpt" \
    --seed 42 \
    --world_idx_list {670..679} \
    --auto_exit \
    --max_duration 10 \
    --save_rollout \
    --result_filename "$eval_dir/baseline_trainseed${train_seed}_I6.yaml" \
    --no_plot \
    --no_render

  python robo_manip_baselines/bin/Rollout.py \
    DiffusionPolicy MujocoUR5eLiftingi_I7 \
    --demo_name "MujocoUR5eLiftingi_I7_baseline_trainseed${train_seed}" \
    --checkpoint "robo_manip_baselines/checkpoint/DiffusionPolicy/LiftingAB_B_only_Baseline_seed${train_seed}/policy_last.ckpt" \
    --seed 42 \
    --world_idx_list {770..779} \
    --auto_exit \
    --max_duration 10 \
    --save_rollout \
    --result_filename "$eval_dir/baseline_trainseed${train_seed}_I7.yaml" \
    --no_plot \
    --no_render

  for object_id in 0 1 2 4 5 6 7; do
    object_dir="$rmb_dir/WrenchPredObject${object_id}"
    mkdir -p "$object_dir"
    find robo_manip_baselines/dataset \
      -mindepth 1 \
      -maxdepth 1 \
      -type d \
      -name "RolloutDiffusionPolicy_MujocoUR5eLiftingi_I${object_id}_baseline_trainseed${train_seed}_20*" \
      -newer "$rollout_start_marker" \
      -exec mv -- {} "$object_dir" \;
  done
done

# Evaluate the 210 saved RMB episodes: lift at least 10 cm and tilt less than
# 7.5 degrees at the final state.
python robo_manip_baselines/misc/futureimagination/AnalyzeLiftingSuccess.py \
  "$rmb_dir" \
  --output_dir "$eval_dir" \
  --output_prefix baseline_training_seed_42_52_62 \
  --expected_episode_count 210
