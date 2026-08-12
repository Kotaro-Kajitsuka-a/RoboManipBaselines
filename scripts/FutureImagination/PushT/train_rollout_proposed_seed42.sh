#!/usr/bin/env bash
set -e

# Run from the repository root.

dataset_dir=robo_manip_baselines/dataset/DatasetMujocoXarm7Pusht_onlinepb2_object0_wp4seed42_lr2p5e2_wrench0p01_20260810
checkpoint_dir=robo_manip_baselines/checkpoint/DiffusionPolicy/PushTPB2/online_seed42
baseline_checkpoint=robo_manip_baselines/checkpoint/DiffusionPolicy/PushTBaseline/seed42/policy_last.ckpt
wp4_checkpoint=robo_manip_baselines/checkpoint/WrenchPredictor4/PushT/policy_best.ckpt
eval_dir=robo_manip_baselines/dataset/tests/FutureImagination/PushTPB2/online_seed42

# Add the causal online-PB trajectories used to train Diffusion Policy.
python robo_manip_baselines/policy/wrench_predictor4_online/AddOnlinePbToDataset.py \
  "$dataset_dir" \
  0 \
  --checkpoint "$wp4_checkpoint" \
  --lr 2.5e-2 \
  --wrench_loss_weight 0.01 \
  --overwrite

# Train the proposed Diffusion Policy with seed 42.
python robo_manip_baselines/bin/Train.py DiffusionPolicy \
  --dataset_dir "$dataset_dir" \
  --checkpoint_dir "$checkpoint_dir" \
  --camera_names \
  --state_keys measured_eef_pose measured_tblock_pose material_property \
  --action_keys command_eef_pose \
  --train_ratio 1.0 \
  --val_ratio 0.01 \
  --seed 42

mkdir -p "$eval_dir"

# Evaluate Object0.
python robo_manip_baselines/bin/Rollout.py DiffusionPolicyOnlinePb MujocoXarm7Pusht_T0 \
  --demo_name PushTPB2Online_seed42_T0 \
  --checkpoint "$checkpoint_dir/policy_last.ckpt" \
  --wp4_checkpoint "$wp4_checkpoint" \
  --initial_object_id 0 \
  --online_pb_lr 2.5e-2 \
  --wrench_loss_weight 0.01 \
  --world_idx_list {70..79} \
  --seed 42 \
  --auto_exit \
  --max_duration 22 \
  --result_filename "$eval_dir/proposed_T0.yaml" \
  --no_plot \
  --no_render

# Evaluate Object1.
python robo_manip_baselines/bin/Rollout.py DiffusionPolicyOnlinePb MujocoXarm7Pusht_T1 \
  --demo_name PushTPB2Online_seed42_T1 \
  --checkpoint "$checkpoint_dir/policy_last.ckpt" \
  --wp4_checkpoint "$wp4_checkpoint" \
  --initial_object_id 0 \
  --online_pb_lr 2.5e-2 \
  --wrench_loss_weight 0.01 \
  --world_idx_list {170..179} \
  --seed 42 \
  --auto_exit \
  --max_duration 22 \
  --result_filename "$eval_dir/proposed_T1.yaml" \
  --no_plot \
  --no_render

# Evaluate Object2.
python robo_manip_baselines/bin/Rollout.py DiffusionPolicyOnlinePb MujocoXarm7Pusht_T2 \
  --demo_name PushTPB2Online_seed42_T2 \
  --checkpoint "$checkpoint_dir/policy_last.ckpt" \
  --wp4_checkpoint "$wp4_checkpoint" \
  --initial_object_id 0 \
  --online_pb_lr 2.5e-2 \
  --wrench_loss_weight 0.01 \
  --world_idx_list {270..279} \
  --seed 42 \
  --auto_exit \
  --max_duration 22 \
  --result_filename "$eval_dir/proposed_T2.yaml" \
  --no_plot \
  --no_render

# Evaluate Object3.
python robo_manip_baselines/bin/Rollout.py DiffusionPolicyOnlinePb MujocoXarm7Pusht_T3 \
  --demo_name PushTPB2Online_seed42_T3 \
  --checkpoint "$checkpoint_dir/policy_last.ckpt" \
  --wp4_checkpoint "$wp4_checkpoint" \
  --initial_object_id 0 \
  --online_pb_lr 2.5e-2 \
  --wrench_loss_weight 0.01 \
  --world_idx_list {370..379} \
  --seed 42 \
  --auto_exit \
  --max_duration 22 \
  --result_filename "$eval_dir/proposed_T3.yaml" \
  --no_plot \
  --no_render

# Evaluate Object4.
python robo_manip_baselines/bin/Rollout.py DiffusionPolicyOnlinePb MujocoXarm7Pusht_T4 \
  --demo_name PushTPB2Online_seed42_T4 \
  --checkpoint "$checkpoint_dir/policy_last.ckpt" \
  --wp4_checkpoint "$wp4_checkpoint" \
  --initial_object_id 0 \
  --online_pb_lr 2.5e-2 \
  --wrench_loss_weight 0.01 \
  --world_idx_list {470..479} \
  --seed 42 \
  --auto_exit \
  --max_duration 22 \
  --result_filename "$eval_dir/proposed_T4.yaml" \
  --no_plot \
  --no_render

# Evaluate the baseline on Object0.
python robo_manip_baselines/bin/Rollout.py DiffusionPolicy MujocoXarm7Pusht_T0 \
  --checkpoint "$baseline_checkpoint" \
  --world_idx_list {70..79} \
  --seed 42 \
  --auto_exit \
  --max_duration 22 \
  --result_filename "$eval_dir/baseline_T0.yaml" \
  --no_plot \
  --no_render

# Evaluate the baseline on Object1.
python robo_manip_baselines/bin/Rollout.py DiffusionPolicy MujocoXarm7Pusht_T1 \
  --checkpoint "$baseline_checkpoint" \
  --world_idx_list {170..179} \
  --seed 42 \
  --auto_exit \
  --max_duration 22 \
  --result_filename "$eval_dir/baseline_T1.yaml" \
  --no_plot \
  --no_render

# Evaluate the baseline on Object2.
python robo_manip_baselines/bin/Rollout.py DiffusionPolicy MujocoXarm7Pusht_T2 \
  --checkpoint "$baseline_checkpoint" \
  --world_idx_list {270..279} \
  --seed 42 \
  --auto_exit \
  --max_duration 22 \
  --result_filename "$eval_dir/baseline_T2.yaml" \
  --no_plot \
  --no_render

# Evaluate the baseline on Object3.
python robo_manip_baselines/bin/Rollout.py DiffusionPolicy MujocoXarm7Pusht_T3 \
  --checkpoint "$baseline_checkpoint" \
  --world_idx_list {370..379} \
  --seed 42 \
  --auto_exit \
  --max_duration 22 \
  --result_filename "$eval_dir/baseline_T3.yaml" \
  --no_plot \
  --no_render

# Evaluate the baseline on Object4.
python robo_manip_baselines/bin/Rollout.py DiffusionPolicy MujocoXarm7Pusht_T4 \
  --checkpoint "$baseline_checkpoint" \
  --world_idx_list {470..479} \
  --seed 42 \
  --auto_exit \
  --max_duration 22 \
  --result_filename "$eval_dir/baseline_T4.yaml" \
  --no_plot \
  --no_render
