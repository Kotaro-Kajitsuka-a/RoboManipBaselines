#!/usr/bin/env bash
set -e

# Evaluate the three existing seed-42 policies and save RMB data with camera MP4s.
# Run from the repository root.

baseline_checkpoint=robo_manip_baselines/checkpoint/DiffusionPolicy/PushTBaseline/seed42/policy_last.ckpt
proposed_checkpoint=robo_manip_baselines/checkpoint/DiffusionPolicy/PushTPB2/online_seed42/policy_last.ckpt
constant_checkpoint=robo_manip_baselines/checkpoint/DiffusionPolicy/PushTPB2/constant_seed42/policy_last.ckpt
wp4_checkpoint=robo_manip_baselines/checkpoint/WrenchPredictor4/PushT/policy_best.ckpt
eval_dir=robo_manip_baselines/dataset/tests/FutureImagination/PushTPB2/save_rollout_seed42

mkdir -p "$eval_dir"

# Baseline.
python robo_manip_baselines/bin/Rollout.py DiffusionPolicy MujocoXarm7Pusht_T0 \
  --demo_name PushTBaseline_seed42_T0 \
  --checkpoint "$baseline_checkpoint" \
  --world_idx_list {70..79} \
  --seed 42 \
  --auto_exit \
  --max_duration 22 \
  --result_filename "$eval_dir/baseline_T0.yaml" \
  --save_rollout \
  --no_plot \
  --no_render

python robo_manip_baselines/bin/Rollout.py DiffusionPolicy MujocoXarm7Pusht_T1 \
  --demo_name PushTBaseline_seed42_T1 \
  --checkpoint "$baseline_checkpoint" \
  --world_idx_list {170..179} \
  --seed 42 \
  --auto_exit \
  --max_duration 22 \
  --result_filename "$eval_dir/baseline_T1.yaml" \
  --save_rollout \
  --no_plot \
  --no_render

python robo_manip_baselines/bin/Rollout.py DiffusionPolicy MujocoXarm7Pusht_T2 \
  --demo_name PushTBaseline_seed42_T2 \
  --checkpoint "$baseline_checkpoint" \
  --world_idx_list {270..279} \
  --seed 42 \
  --auto_exit \
  --max_duration 22 \
  --result_filename "$eval_dir/baseline_T2.yaml" \
  --save_rollout \
  --no_plot \
  --no_render

python robo_manip_baselines/bin/Rollout.py DiffusionPolicy MujocoXarm7Pusht_T3 \
  --demo_name PushTBaseline_seed42_T3 \
  --checkpoint "$baseline_checkpoint" \
  --world_idx_list {370..379} \
  --seed 42 \
  --auto_exit \
  --max_duration 22 \
  --result_filename "$eval_dir/baseline_T3.yaml" \
  --save_rollout \
  --no_plot \
  --no_render

python robo_manip_baselines/bin/Rollout.py DiffusionPolicy MujocoXarm7Pusht_T4 \
  --demo_name PushTBaseline_seed42_T4 \
  --checkpoint "$baseline_checkpoint" \
  --world_idx_list {470..479} \
  --seed 42 \
  --auto_exit \
  --max_duration 22 \
  --result_filename "$eval_dir/baseline_T4.yaml" \
  --save_rollout \
  --no_plot \
  --no_render

# Proposed online PB.
python robo_manip_baselines/bin/Rollout.py DiffusionPolicyOnlinePb MujocoXarm7Pusht_T0 \
  --demo_name PushTPB2Online_seed42_T0 \
  --checkpoint "$proposed_checkpoint" \
  --wp4_checkpoint "$wp4_checkpoint" \
  --initial_object_id 0 \
  --online_pb_lr 2.5e-2 \
  --wrench_loss_weight 0.01 \
  --world_idx_list {70..79} \
  --seed 42 \
  --auto_exit \
  --max_duration 22 \
  --result_filename "$eval_dir/proposed_T0.yaml" \
  --save_rollout \
  --no_plot \
  --no_render

python robo_manip_baselines/bin/Rollout.py DiffusionPolicyOnlinePb MujocoXarm7Pusht_T1 \
  --demo_name PushTPB2Online_seed42_T1 \
  --checkpoint "$proposed_checkpoint" \
  --wp4_checkpoint "$wp4_checkpoint" \
  --initial_object_id 0 \
  --online_pb_lr 2.5e-2 \
  --wrench_loss_weight 0.01 \
  --world_idx_list {170..179} \
  --seed 42 \
  --auto_exit \
  --max_duration 22 \
  --result_filename "$eval_dir/proposed_T1.yaml" \
  --save_rollout \
  --no_plot \
  --no_render

python robo_manip_baselines/bin/Rollout.py DiffusionPolicyOnlinePb MujocoXarm7Pusht_T2 \
  --demo_name PushTPB2Online_seed42_T2 \
  --checkpoint "$proposed_checkpoint" \
  --wp4_checkpoint "$wp4_checkpoint" \
  --initial_object_id 0 \
  --online_pb_lr 2.5e-2 \
  --wrench_loss_weight 0.01 \
  --world_idx_list {270..279} \
  --seed 42 \
  --auto_exit \
  --max_duration 22 \
  --result_filename "$eval_dir/proposed_T2.yaml" \
  --save_rollout \
  --no_plot \
  --no_render

python robo_manip_baselines/bin/Rollout.py DiffusionPolicyOnlinePb MujocoXarm7Pusht_T3 \
  --demo_name PushTPB2Online_seed42_T3 \
  --checkpoint "$proposed_checkpoint" \
  --wp4_checkpoint "$wp4_checkpoint" \
  --initial_object_id 0 \
  --online_pb_lr 2.5e-2 \
  --wrench_loss_weight 0.01 \
  --world_idx_list {370..379} \
  --seed 42 \
  --auto_exit \
  --max_duration 22 \
  --result_filename "$eval_dir/proposed_T3.yaml" \
  --save_rollout \
  --no_plot \
  --no_render

python robo_manip_baselines/bin/Rollout.py DiffusionPolicyOnlinePb MujocoXarm7Pusht_T4 \
  --demo_name PushTPB2Online_seed42_T4 \
  --checkpoint "$proposed_checkpoint" \
  --wp4_checkpoint "$wp4_checkpoint" \
  --initial_object_id 0 \
  --online_pb_lr 2.5e-2 \
  --wrench_loss_weight 0.01 \
  --world_idx_list {470..479} \
  --seed 42 \
  --auto_exit \
  --max_duration 22 \
  --result_filename "$eval_dir/proposed_T4.yaml" \
  --save_rollout \
  --no_plot \
  --no_render

# Constant-PB-trained Diffusion Policy with the same online PB adaptation.
python robo_manip_baselines/bin/Rollout.py DiffusionPolicyOnlinePb MujocoXarm7Pusht_T0 \
  --demo_name PushTPB2Constant_seed42_T0 \
  --checkpoint "$constant_checkpoint" \
  --wp4_checkpoint "$wp4_checkpoint" \
  --initial_object_id 0 \
  --online_pb_lr 2.5e-2 \
  --wrench_loss_weight 0.01 \
  --world_idx_list {70..79} \
  --seed 42 \
  --auto_exit \
  --max_duration 22 \
  --result_filename "$eval_dir/constant_T0.yaml" \
  --save_rollout \
  --no_plot \
  --no_render

python robo_manip_baselines/bin/Rollout.py DiffusionPolicyOnlinePb MujocoXarm7Pusht_T1 \
  --demo_name PushTPB2Constant_seed42_T1 \
  --checkpoint "$constant_checkpoint" \
  --wp4_checkpoint "$wp4_checkpoint" \
  --initial_object_id 0 \
  --online_pb_lr 2.5e-2 \
  --wrench_loss_weight 0.01 \
  --world_idx_list {170..179} \
  --seed 42 \
  --auto_exit \
  --max_duration 22 \
  --result_filename "$eval_dir/constant_T1.yaml" \
  --save_rollout \
  --no_plot \
  --no_render

python robo_manip_baselines/bin/Rollout.py DiffusionPolicyOnlinePb MujocoXarm7Pusht_T2 \
  --demo_name PushTPB2Constant_seed42_T2 \
  --checkpoint "$constant_checkpoint" \
  --wp4_checkpoint "$wp4_checkpoint" \
  --initial_object_id 0 \
  --online_pb_lr 2.5e-2 \
  --wrench_loss_weight 0.01 \
  --world_idx_list {270..279} \
  --seed 42 \
  --auto_exit \
  --max_duration 22 \
  --result_filename "$eval_dir/constant_T2.yaml" \
  --save_rollout \
  --no_plot \
  --no_render

python robo_manip_baselines/bin/Rollout.py DiffusionPolicyOnlinePb MujocoXarm7Pusht_T3 \
  --demo_name PushTPB2Constant_seed42_T3 \
  --checkpoint "$constant_checkpoint" \
  --wp4_checkpoint "$wp4_checkpoint" \
  --initial_object_id 0 \
  --online_pb_lr 2.5e-2 \
  --wrench_loss_weight 0.01 \
  --world_idx_list {370..379} \
  --seed 42 \
  --auto_exit \
  --max_duration 22 \
  --result_filename "$eval_dir/constant_T3.yaml" \
  --save_rollout \
  --no_plot \
  --no_render

python robo_manip_baselines/bin/Rollout.py DiffusionPolicyOnlinePb MujocoXarm7Pusht_T4 \
  --demo_name PushTPB2Constant_seed42_T4 \
  --checkpoint "$constant_checkpoint" \
  --wp4_checkpoint "$wp4_checkpoint" \
  --initial_object_id 0 \
  --online_pb_lr 2.5e-2 \
  --wrench_loss_weight 0.01 \
  --world_idx_list {470..479} \
  --seed 42 \
  --auto_exit \
  --max_duration 22 \
  --result_filename "$eval_dir/constant_T4.yaml" \
  --save_rollout \
  --no_plot \
  --no_render
