#!/usr/bin/env bash
set -e

# Evaluate the three seed-42 policies and save RMB data with camera MP4s.
# Run from the repository root after train_seed42.sh.

baseline_checkpoint=robo_manip_baselines/checkpoint/DiffusionPolicy/AdmittancePushT/baseline_seed42/policy_last.ckpt
online_checkpoint=robo_manip_baselines/checkpoint/DiffusionPolicy/AdmittancePushT/online_seed42/policy_last.ckpt
constant_checkpoint=robo_manip_baselines/checkpoint/DiffusionPolicy/AdmittancePushT/constant_seed42/policy_last.ckpt
wp4_checkpoint=robo_manip_baselines/checkpoint/WrenchPredictor4/AdmittancePushT/policy_best.ckpt
eval_dir=robo_manip_baselines/dataset/tests/FutureImagination/AdmittancePushT/save_rollout_seed42
rmb_dir="$eval_dir/rmb"

mkdir -p "$rmb_dir"
rollout_start_marker="$eval_dir/.rollout_start"
touch "$rollout_start_marker"

# Baseline.
python robo_manip_baselines/bin/Rollout.py DiffusionPolicy MujocoXarm7AdmittancePusht_T0 \
  --demo_name AdmittancePushTBaseline_seed42_T0 \
  --checkpoint "$baseline_checkpoint" \
  --world_idx_list {70..79} \
  --seed 42 \
  --auto_exit \
  --max_duration 22 \
  --result_filename "$eval_dir/baseline_T0.yaml" \
  --save_rollout \
  --no_plot \
  --no_render

python robo_manip_baselines/bin/Rollout.py DiffusionPolicy MujocoXarm7AdmittancePusht_T1 \
  --demo_name AdmittancePushTBaseline_seed42_T1 \
  --checkpoint "$baseline_checkpoint" \
  --world_idx_list {170..179} \
  --seed 42 \
  --auto_exit \
  --max_duration 22 \
  --result_filename "$eval_dir/baseline_T1.yaml" \
  --save_rollout \
  --no_plot \
  --no_render

python robo_manip_baselines/bin/Rollout.py DiffusionPolicy MujocoXarm7AdmittancePusht_T2 \
  --demo_name AdmittancePushTBaseline_seed42_T2 \
  --checkpoint "$baseline_checkpoint" \
  --world_idx_list {270..279} \
  --seed 42 \
  --auto_exit \
  --max_duration 22 \
  --result_filename "$eval_dir/baseline_T2.yaml" \
  --save_rollout \
  --no_plot \
  --no_render

python robo_manip_baselines/bin/Rollout.py DiffusionPolicy MujocoXarm7AdmittancePusht_T3 \
  --demo_name AdmittancePushTBaseline_seed42_T3 \
  --checkpoint "$baseline_checkpoint" \
  --world_idx_list {370..379} \
  --seed 42 \
  --auto_exit \
  --max_duration 22 \
  --result_filename "$eval_dir/baseline_T3.yaml" \
  --save_rollout \
  --no_plot \
  --no_render

# Proposed online PB.
python robo_manip_baselines/bin/Rollout.py DiffusionPolicyOnlinePb MujocoXarm7AdmittancePusht_T0 \
  --demo_name AdmittancePushTOnline_seed42_T0 \
  --checkpoint "$online_checkpoint" \
  --wp4_checkpoint "$wp4_checkpoint" \
  --initial_object_id 0 \
  --online_pb_lr 2.5e-2 \
  --wrench_loss_weight 0.01 \
  --world_idx_list {70..79} \
  --seed 42 \
  --auto_exit \
  --max_duration 22 \
  --result_filename "$eval_dir/online_T0.yaml" \
  --save_rollout \
  --no_plot \
  --no_render

python robo_manip_baselines/bin/Rollout.py DiffusionPolicyOnlinePb MujocoXarm7AdmittancePusht_T1 \
  --demo_name AdmittancePushTOnline_seed42_T1 \
  --checkpoint "$online_checkpoint" \
  --wp4_checkpoint "$wp4_checkpoint" \
  --initial_object_id 0 \
  --online_pb_lr 2.5e-2 \
  --wrench_loss_weight 0.01 \
  --world_idx_list {170..179} \
  --seed 42 \
  --auto_exit \
  --max_duration 22 \
  --result_filename "$eval_dir/online_T1.yaml" \
  --save_rollout \
  --no_plot \
  --no_render

python robo_manip_baselines/bin/Rollout.py DiffusionPolicyOnlinePb MujocoXarm7AdmittancePusht_T2 \
  --demo_name AdmittancePushTOnline_seed42_T2 \
  --checkpoint "$online_checkpoint" \
  --wp4_checkpoint "$wp4_checkpoint" \
  --initial_object_id 0 \
  --online_pb_lr 2.5e-2 \
  --wrench_loss_weight 0.01 \
  --world_idx_list {270..279} \
  --seed 42 \
  --auto_exit \
  --max_duration 22 \
  --result_filename "$eval_dir/online_T2.yaml" \
  --save_rollout \
  --no_plot \
  --no_render

python robo_manip_baselines/bin/Rollout.py DiffusionPolicyOnlinePb MujocoXarm7AdmittancePusht_T3 \
  --demo_name AdmittancePushTOnline_seed42_T3 \
  --checkpoint "$online_checkpoint" \
  --wp4_checkpoint "$wp4_checkpoint" \
  --initial_object_id 0 \
  --online_pb_lr 2.5e-2 \
  --wrench_loss_weight 0.01 \
  --world_idx_list {370..379} \
  --seed 42 \
  --auto_exit \
  --max_duration 22 \
  --result_filename "$eval_dir/online_T3.yaml" \
  --save_rollout \
  --no_plot \
  --no_render

# Constant-PB-trained Diffusion Policy with online PB adaptation.
python robo_manip_baselines/bin/Rollout.py DiffusionPolicyOnlinePb MujocoXarm7AdmittancePusht_T0 \
  --demo_name AdmittancePushTConstant_seed42_T0 \
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

python robo_manip_baselines/bin/Rollout.py DiffusionPolicyOnlinePb MujocoXarm7AdmittancePusht_T1 \
  --demo_name AdmittancePushTConstant_seed42_T1 \
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

python robo_manip_baselines/bin/Rollout.py DiffusionPolicyOnlinePb MujocoXarm7AdmittancePusht_T2 \
  --demo_name AdmittancePushTConstant_seed42_T2 \
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

python robo_manip_baselines/bin/Rollout.py DiffusionPolicyOnlinePb MujocoXarm7AdmittancePusht_T3 \
  --demo_name AdmittancePushTConstant_seed42_T3 \
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

find robo_manip_baselines/dataset \
  -mindepth 1 \
  -maxdepth 1 \
  -type d \
  \( \
    -name "RolloutDiffusionPolicy_AdmittancePushTBaseline_seed42_T[0-3]_20*" \
    -o -name "RolloutDiffusionPolicyOnlinePb_AdmittancePushTOnline_seed42_T[0-3]_20*" \
    -o -name "RolloutDiffusionPolicyOnlinePb_AdmittancePushTConstant_seed42_T[0-3]_20*" \
  \) \
  -newer "$rollout_start_marker" \
  -exec mv -- {} "$rmb_dir" \;
