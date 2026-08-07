#!/usr/bin/env bash
set -e

# Run from the repository root after train_baseline_seed42_52_62.sh.
# Evaluate ordinary Diffusion Policy without online PB adaptation.
mkdir -p robo_manip_baselines/dataset/tests/FutureImagination/LiftingAB_DP_baseline_seed_eval_20260807

# Train seed 42, rollout seed 42.
python robo_manip_baselines/bin/Rollout.py \
  DiffusionPolicy MujocoUR5eLiftingi_I0 \
  --checkpoint robo_manip_baselines/checkpoint/DiffusionPolicy/LiftingAB_B_only_Baseline_seed42/policy_last.ckpt \
  --world_idx_list {70..79} \
  --seed 42 \
  --auto_exit \
  --max_duration 10 \
  --save_rollout \
  --result_filename robo_manip_baselines/dataset/tests/FutureImagination/LiftingAB_DP_baseline_seed_eval_20260807/baseline_trainseed42_rollseed42_I0.yaml \
  --no_plot

python robo_manip_baselines/bin/Rollout.py \
  DiffusionPolicy MujocoUR5eLiftingi_I1 \
  --checkpoint robo_manip_baselines/checkpoint/DiffusionPolicy/LiftingAB_B_only_Baseline_seed42/policy_last.ckpt \
  --world_idx_list {170..179} \
  --seed 42 \
  --auto_exit \
  --max_duration 10 \
  --save_rollout \
  --result_filename robo_manip_baselines/dataset/tests/FutureImagination/LiftingAB_DP_baseline_seed_eval_20260807/baseline_trainseed42_rollseed42_I1.yaml \
  --no_plot

python robo_manip_baselines/bin/Rollout.py \
  DiffusionPolicy MujocoUR5eLiftingi_I2 \
  --checkpoint robo_manip_baselines/checkpoint/DiffusionPolicy/LiftingAB_B_only_Baseline_seed42/policy_last.ckpt \
  --world_idx_list {270..279} \
  --seed 42 \
  --auto_exit \
  --max_duration 10 \
  --save_rollout \
  --result_filename robo_manip_baselines/dataset/tests/FutureImagination/LiftingAB_DP_baseline_seed_eval_20260807/baseline_trainseed42_rollseed42_I2.yaml \
  --no_plot

# Train seed 52, rollout seed 42.
python robo_manip_baselines/bin/Rollout.py \
  DiffusionPolicy MujocoUR5eLiftingi_I0 \
  --checkpoint robo_manip_baselines/checkpoint/DiffusionPolicy/LiftingAB_B_only_Baseline_seed52/policy_last.ckpt \
  --world_idx_list {70..79} \
  --seed 42 \
  --auto_exit \
  --max_duration 10 \
  --save_rollout \
  --result_filename robo_manip_baselines/dataset/tests/FutureImagination/LiftingAB_DP_baseline_seed_eval_20260807/baseline_trainseed52_rollseed42_I0.yaml \
  --no_plot

python robo_manip_baselines/bin/Rollout.py \
  DiffusionPolicy MujocoUR5eLiftingi_I1 \
  --checkpoint robo_manip_baselines/checkpoint/DiffusionPolicy/LiftingAB_B_only_Baseline_seed52/policy_last.ckpt \
  --world_idx_list {170..179} \
  --seed 42 \
  --auto_exit \
  --max_duration 10 \
  --save_rollout \
  --result_filename robo_manip_baselines/dataset/tests/FutureImagination/LiftingAB_DP_baseline_seed_eval_20260807/baseline_trainseed52_rollseed42_I1.yaml \
  --no_plot

python robo_manip_baselines/bin/Rollout.py \
  DiffusionPolicy MujocoUR5eLiftingi_I2 \
  --checkpoint robo_manip_baselines/checkpoint/DiffusionPolicy/LiftingAB_B_only_Baseline_seed52/policy_last.ckpt \
  --world_idx_list {270..279} \
  --seed 42 \
  --auto_exit \
  --max_duration 10 \
  --save_rollout \
  --result_filename robo_manip_baselines/dataset/tests/FutureImagination/LiftingAB_DP_baseline_seed_eval_20260807/baseline_trainseed52_rollseed42_I2.yaml \
  --no_plot

# Train seed 62, rollout seed 42.
python robo_manip_baselines/bin/Rollout.py \
  DiffusionPolicy MujocoUR5eLiftingi_I0 \
  --checkpoint robo_manip_baselines/checkpoint/DiffusionPolicy/LiftingAB_B_only_Baseline_seed62/policy_last.ckpt \
  --world_idx_list {70..79} \
  --seed 42 \
  --auto_exit \
  --max_duration 10 \
  --save_rollout \
  --result_filename robo_manip_baselines/dataset/tests/FutureImagination/LiftingAB_DP_baseline_seed_eval_20260807/baseline_trainseed62_rollseed42_I0.yaml \
  --no_plot

python robo_manip_baselines/bin/Rollout.py \
  DiffusionPolicy MujocoUR5eLiftingi_I1 \
  --checkpoint robo_manip_baselines/checkpoint/DiffusionPolicy/LiftingAB_B_only_Baseline_seed62/policy_last.ckpt \
  --world_idx_list {170..179} \
  --seed 42 \
  --auto_exit \
  --max_duration 10 \
  --save_rollout \
  --result_filename robo_manip_baselines/dataset/tests/FutureImagination/LiftingAB_DP_baseline_seed_eval_20260807/baseline_trainseed62_rollseed42_I1.yaml \
  --no_plot

python robo_manip_baselines/bin/Rollout.py \
  DiffusionPolicy MujocoUR5eLiftingi_I2 \
  --checkpoint robo_manip_baselines/checkpoint/DiffusionPolicy/LiftingAB_B_only_Baseline_seed62/policy_last.ckpt \
  --world_idx_list {270..279} \
  --seed 42 \
  --auto_exit \
  --max_duration 10 \
  --save_rollout \
  --result_filename robo_manip_baselines/dataset/tests/FutureImagination/LiftingAB_DP_baseline_seed_eval_20260807/baseline_trainseed62_rollseed42_I2.yaml \
  --no_plot

# Evaluate the 90 saved RMB episodes: lift at least 10 cm and tilt less than
# 7.5 degrees at the final state.
python robo_manip_baselines/misc/futureimagination/AnalyzeLiftingSuccess.py \
  robo_manip_baselines/dataset/RolloutDiffusionPolicy_MujocoUR5eLiftingi_I0_20??????_?????? \
  robo_manip_baselines/dataset/RolloutDiffusionPolicy_MujocoUR5eLiftingi_I1_20??????_?????? \
  robo_manip_baselines/dataset/RolloutDiffusionPolicy_MujocoUR5eLiftingi_I2_20??????_?????? \
  --output_dir robo_manip_baselines/dataset/tests/FutureImagination/LiftingAB_DP_baseline_seed_eval_20260807 \
  --output_prefix baseline_training_seed_42_52_62_rollseed42 \
  --expected_episode_count 90
