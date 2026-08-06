#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

eval_dir=robo_manip_baselines/dataset/tests/FutureImagination/DatasetMujocoXarm7Pusht_eval_rollseed42_52_62
wp4_checkpoint=robo_manip_baselines/checkpoint/WrenchPredictor4/DatasetMujocoXarm7Pusht_WrenchPredObject0-4_100train_pb1_mlp_only_pose9_h16_e500_seed42_20260805/policy_best.ckpt
seed42_checkpoint=robo_manip_baselines/checkpoint/DiffusionPolicy/DatasetMujocoXarm7Pusht_100train_onlinepb_object0_wp4seed42_state_eef_tblock_pb1_cnn_ddim_h16_e500_seed42_20260805/policy_last.ckpt
seed52_checkpoint=robo_manip_baselines/checkpoint/DiffusionPolicy/DatasetMujocoXarm7Pusht_100train_onlinepb_object0_wp4seed42_state_eef_tblock_pb1_cnn_ddim_h16_e500_seed52_20260805/policy_last.ckpt
seed62_checkpoint=robo_manip_baselines/checkpoint/DiffusionPolicy/DatasetMujocoXarm7Pusht_100train_onlinepb_object0_wp4seed42_state_eef_tblock_pb1_cnn_ddim_h16_e500_seed62_20260805/policy_last.ckpt

mkdir -p "$eval_dir" &&
uv run python robo_manip_baselines/bin/Rollout.py DiffusionPolicyOnlinePb MujocoXarm7Pusht_T0 \
  --checkpoint "$seed42_checkpoint" \
  --wp4_checkpoint "$wp4_checkpoint" \
  --initial_object_id 0 \
  --online_pb_lr 6e-3 \
  --world_idx_list {70..79} \
  --seed 42 \
  --auto_exit \
  --max_duration 22 \
  --save_rollout \
  --result_filename "$eval_dir/proposed_trainseed42_rollseed42_T0.yaml" \
  --no_plot \
  --no_render &&
uv run python robo_manip_baselines/bin/Rollout.py DiffusionPolicyOnlinePb MujocoXarm7Pusht_T1 \
  --checkpoint "$seed42_checkpoint" \
  --wp4_checkpoint "$wp4_checkpoint" \
  --initial_object_id 0 \
  --online_pb_lr 6e-3 \
  --world_idx_list {170..179} \
  --seed 42 \
  --auto_exit \
  --max_duration 22 \
  --save_rollout \
  --result_filename "$eval_dir/proposed_trainseed42_rollseed42_T1.yaml" \
  --no_plot \
  --no_render &&
uv run python robo_manip_baselines/bin/Rollout.py DiffusionPolicyOnlinePb MujocoXarm7Pusht_T2 \
  --checkpoint "$seed42_checkpoint" \
  --wp4_checkpoint "$wp4_checkpoint" \
  --initial_object_id 0 \
  --online_pb_lr 6e-3 \
  --world_idx_list {270..279} \
  --seed 42 \
  --auto_exit \
  --max_duration 22 \
  --save_rollout \
  --result_filename "$eval_dir/proposed_trainseed42_rollseed42_T2.yaml" \
  --no_plot \
  --no_render &&
uv run python robo_manip_baselines/bin/Rollout.py DiffusionPolicyOnlinePb MujocoXarm7Pusht_T3 \
  --checkpoint "$seed42_checkpoint" \
  --wp4_checkpoint "$wp4_checkpoint" \
  --initial_object_id 0 \
  --online_pb_lr 6e-3 \
  --world_idx_list {370..379} \
  --seed 42 \
  --auto_exit \
  --max_duration 22 \
  --save_rollout \
  --result_filename "$eval_dir/proposed_trainseed42_rollseed42_T3.yaml" \
  --no_plot \
  --no_render &&
uv run python robo_manip_baselines/bin/Rollout.py DiffusionPolicyOnlinePb MujocoXarm7Pusht_T4 \
  --checkpoint "$seed42_checkpoint" \
  --wp4_checkpoint "$wp4_checkpoint" \
  --initial_object_id 0 \
  --online_pb_lr 6e-3 \
  --world_idx_list {470..479} \
  --seed 42 \
  --auto_exit \
  --max_duration 22 \
  --save_rollout \
  --result_filename "$eval_dir/proposed_trainseed42_rollseed42_T4.yaml" \
  --no_plot \
  --no_render &&
uv run python robo_manip_baselines/bin/Rollout.py DiffusionPolicyOnlinePb MujocoXarm7Pusht_T0 \
  --checkpoint "$seed52_checkpoint" \
  --wp4_checkpoint "$wp4_checkpoint" \
  --initial_object_id 0 \
  --online_pb_lr 6e-3 \
  --world_idx_list {70..79} \
  --seed 42 \
  --auto_exit \
  --max_duration 22 \
  --save_rollout \
  --result_filename "$eval_dir/proposed_trainseed52_rollseed42_T0.yaml" \
  --no_plot \
  --no_render &&
uv run python robo_manip_baselines/bin/Rollout.py DiffusionPolicyOnlinePb MujocoXarm7Pusht_T1 \
  --checkpoint "$seed52_checkpoint" \
  --wp4_checkpoint "$wp4_checkpoint" \
  --initial_object_id 0 \
  --online_pb_lr 6e-3 \
  --world_idx_list {170..179} \
  --seed 42 \
  --auto_exit \
  --max_duration 22 \
  --save_rollout \
  --result_filename "$eval_dir/proposed_trainseed52_rollseed42_T1.yaml" \
  --no_plot \
  --no_render &&
uv run python robo_manip_baselines/bin/Rollout.py DiffusionPolicyOnlinePb MujocoXarm7Pusht_T2 \
  --checkpoint "$seed52_checkpoint" \
  --wp4_checkpoint "$wp4_checkpoint" \
  --initial_object_id 0 \
  --online_pb_lr 6e-3 \
  --world_idx_list {270..279} \
  --seed 42 \
  --auto_exit \
  --max_duration 22 \
  --save_rollout \
  --result_filename "$eval_dir/proposed_trainseed52_rollseed42_T2.yaml" \
  --no_plot \
  --no_render &&
uv run python robo_manip_baselines/bin/Rollout.py DiffusionPolicyOnlinePb MujocoXarm7Pusht_T3 \
  --checkpoint "$seed52_checkpoint" \
  --wp4_checkpoint "$wp4_checkpoint" \
  --initial_object_id 0 \
  --online_pb_lr 6e-3 \
  --world_idx_list {370..379} \
  --seed 42 \
  --auto_exit \
  --max_duration 22 \
  --save_rollout \
  --result_filename "$eval_dir/proposed_trainseed52_rollseed42_T3.yaml" \
  --no_plot \
  --no_render &&
uv run python robo_manip_baselines/bin/Rollout.py DiffusionPolicyOnlinePb MujocoXarm7Pusht_T4 \
  --checkpoint "$seed52_checkpoint" \
  --wp4_checkpoint "$wp4_checkpoint" \
  --initial_object_id 0 \
  --online_pb_lr 6e-3 \
  --world_idx_list {470..479} \
  --seed 42 \
  --auto_exit \
  --max_duration 22 \
  --save_rollout \
  --result_filename "$eval_dir/proposed_trainseed52_rollseed42_T4.yaml" \
  --no_plot \
  --no_render &&
uv run python robo_manip_baselines/bin/Rollout.py DiffusionPolicyOnlinePb MujocoXarm7Pusht_T0 \
  --checkpoint "$seed62_checkpoint" \
  --wp4_checkpoint "$wp4_checkpoint" \
  --initial_object_id 0 \
  --online_pb_lr 6e-3 \
  --world_idx_list {70..79} \
  --seed 42 \
  --auto_exit \
  --max_duration 22 \
  --save_rollout \
  --result_filename "$eval_dir/proposed_trainseed62_rollseed42_T0.yaml" \
  --no_plot \
  --no_render &&
uv run python robo_manip_baselines/bin/Rollout.py DiffusionPolicyOnlinePb MujocoXarm7Pusht_T1 \
  --checkpoint "$seed62_checkpoint" \
  --wp4_checkpoint "$wp4_checkpoint" \
  --initial_object_id 0 \
  --online_pb_lr 6e-3 \
  --world_idx_list {170..179} \
  --seed 42 \
  --auto_exit \
  --max_duration 22 \
  --save_rollout \
  --result_filename "$eval_dir/proposed_trainseed62_rollseed42_T1.yaml" \
  --no_plot \
  --no_render &&
uv run python robo_manip_baselines/bin/Rollout.py DiffusionPolicyOnlinePb MujocoXarm7Pusht_T2 \
  --checkpoint "$seed62_checkpoint" \
  --wp4_checkpoint "$wp4_checkpoint" \
  --initial_object_id 0 \
  --online_pb_lr 6e-3 \
  --world_idx_list {270..279} \
  --seed 42 \
  --auto_exit \
  --max_duration 22 \
  --save_rollout \
  --result_filename "$eval_dir/proposed_trainseed62_rollseed42_T2.yaml" \
  --no_plot \
  --no_render &&
uv run python robo_manip_baselines/bin/Rollout.py DiffusionPolicyOnlinePb MujocoXarm7Pusht_T3 \
  --checkpoint "$seed62_checkpoint" \
  --wp4_checkpoint "$wp4_checkpoint" \
  --initial_object_id 0 \
  --online_pb_lr 6e-3 \
  --world_idx_list {370..379} \
  --seed 42 \
  --auto_exit \
  --max_duration 22 \
  --save_rollout \
  --result_filename "$eval_dir/proposed_trainseed62_rollseed42_T3.yaml" \
  --no_plot \
  --no_render &&
uv run python robo_manip_baselines/bin/Rollout.py DiffusionPolicyOnlinePb MujocoXarm7Pusht_T4 \
  --checkpoint "$seed62_checkpoint" \
  --wp4_checkpoint "$wp4_checkpoint" \
  --initial_object_id 0 \
  --online_pb_lr 6e-3 \
  --world_idx_list {470..479} \
  --seed 42 \
  --auto_exit \
  --max_duration 22 \
  --save_rollout \
  --result_filename "$eval_dir/proposed_trainseed62_rollseed42_T4.yaml" \
  --no_plot \
  --no_render
