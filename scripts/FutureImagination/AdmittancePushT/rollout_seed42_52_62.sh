#!/usr/bin/env bash
set -e

# Evaluate training seeds 42, 52, and 62 with rollout seed 42.
# Save RMB data with camera MP4s.
# Run from the repository root after train_seed42_52_62.sh.

wp4_checkpoint=robo_manip_baselines/checkpoint/WrenchPredictor4/AdmittancePushT/policy_best.ckpt
eval_dir=robo_manip_baselines/dataset/tests/FutureImagination/AdmittancePushT/rollout_seed42_52_62
rmb_dir="$eval_dir/rmb"

mkdir -p "$rmb_dir"
rollout_start_marker="$eval_dir/.rollout_start"
touch "$rollout_start_marker"

for train_seed in 42 52 62; do
  baseline_checkpoint="robo_manip_baselines/checkpoint/DiffusionPolicy/AdmittancePushT/baseline_seed${train_seed}/policy_last.ckpt"
  online_checkpoint="robo_manip_baselines/checkpoint/DiffusionPolicy/AdmittancePushT/online_seed${train_seed}/policy_last.ckpt"
  constant_checkpoint="robo_manip_baselines/checkpoint/DiffusionPolicy/AdmittancePushT/constant_seed${train_seed}/policy_last.ckpt"

  # Baseline.
  python robo_manip_baselines/bin/Rollout.py DiffusionPolicy MujocoXarm7AdmittancePusht_T0 \
    --demo_name AdmittancePushTBaseline_trainseed${train_seed}_rollseed42_T0 \
    --checkpoint "$baseline_checkpoint" \
    --world_idx_list {70..79} \
    --seed 42 \
    --auto_exit \
    --max_duration 22 \
    --result_filename "$eval_dir/baseline_trainseed${train_seed}_rollseed42_T0.yaml" \
    --save_rollout \
    --no_plot \
    --no_render

  python robo_manip_baselines/bin/Rollout.py DiffusionPolicy MujocoXarm7AdmittancePusht_T1 \
    --demo_name AdmittancePushTBaseline_trainseed${train_seed}_rollseed42_T1 \
    --checkpoint "$baseline_checkpoint" \
    --world_idx_list {170..179} \
    --seed 42 \
    --auto_exit \
    --max_duration 22 \
    --result_filename "$eval_dir/baseline_trainseed${train_seed}_rollseed42_T1.yaml" \
    --save_rollout \
    --no_plot \
    --no_render

  python robo_manip_baselines/bin/Rollout.py DiffusionPolicy MujocoXarm7AdmittancePusht_T2 \
    --demo_name AdmittancePushTBaseline_trainseed${train_seed}_rollseed42_T2 \
    --checkpoint "$baseline_checkpoint" \
    --world_idx_list {270..279} \
    --seed 42 \
    --auto_exit \
    --max_duration 22 \
    --result_filename "$eval_dir/baseline_trainseed${train_seed}_rollseed42_T2.yaml" \
    --save_rollout \
    --no_plot \
    --no_render

  python robo_manip_baselines/bin/Rollout.py DiffusionPolicy MujocoXarm7AdmittancePusht_T3 \
    --demo_name AdmittancePushTBaseline_trainseed${train_seed}_rollseed42_T3 \
    --checkpoint "$baseline_checkpoint" \
    --world_idx_list {370..379} \
    --seed 42 \
    --auto_exit \
    --max_duration 22 \
    --result_filename "$eval_dir/baseline_trainseed${train_seed}_rollseed42_T3.yaml" \
    --save_rollout \
    --no_plot \
    --no_render

  python robo_manip_baselines/bin/Rollout.py DiffusionPolicy MujocoXarm7AdmittancePusht_T4 \
    --demo_name AdmittancePushTBaseline_trainseed${train_seed}_rollseed42_T4 \
    --checkpoint "$baseline_checkpoint" \
    --world_idx_list {470..479} \
    --seed 42 \
    --auto_exit \
    --max_duration 22 \
    --result_filename "$eval_dir/baseline_trainseed${train_seed}_rollseed42_T4.yaml" \
    --save_rollout \
    --no_plot \
    --no_render

  # Proposed online PB.
  python robo_manip_baselines/bin/Rollout.py DiffusionPolicyOnlinePb MujocoXarm7AdmittancePusht_T0 \
    --demo_name AdmittancePushTOnline_trainseed${train_seed}_rollseed42_T0 \
    --checkpoint "$online_checkpoint" \
    --wp4_checkpoint "$wp4_checkpoint" \
    --initial_object_id 0 \
    --online_pb_lr 2.5e-2 \
    --wrench_loss_weight 0.01 \
    --world_idx_list {70..79} \
    --seed 42 \
    --auto_exit \
    --max_duration 22 \
    --result_filename "$eval_dir/online_trainseed${train_seed}_rollseed42_T0.yaml" \
    --save_rollout \
    --no_plot \
    --no_render

  python robo_manip_baselines/bin/Rollout.py DiffusionPolicyOnlinePb MujocoXarm7AdmittancePusht_T1 \
    --demo_name AdmittancePushTOnline_trainseed${train_seed}_rollseed42_T1 \
    --checkpoint "$online_checkpoint" \
    --wp4_checkpoint "$wp4_checkpoint" \
    --initial_object_id 0 \
    --online_pb_lr 2.5e-2 \
    --wrench_loss_weight 0.01 \
    --world_idx_list {170..179} \
    --seed 42 \
    --auto_exit \
    --max_duration 22 \
    --result_filename "$eval_dir/online_trainseed${train_seed}_rollseed42_T1.yaml" \
    --save_rollout \
    --no_plot \
    --no_render

  python robo_manip_baselines/bin/Rollout.py DiffusionPolicyOnlinePb MujocoXarm7AdmittancePusht_T2 \
    --demo_name AdmittancePushTOnline_trainseed${train_seed}_rollseed42_T2 \
    --checkpoint "$online_checkpoint" \
    --wp4_checkpoint "$wp4_checkpoint" \
    --initial_object_id 0 \
    --online_pb_lr 2.5e-2 \
    --wrench_loss_weight 0.01 \
    --world_idx_list {270..279} \
    --seed 42 \
    --auto_exit \
    --max_duration 22 \
    --result_filename "$eval_dir/online_trainseed${train_seed}_rollseed42_T2.yaml" \
    --save_rollout \
    --no_plot \
    --no_render

  python robo_manip_baselines/bin/Rollout.py DiffusionPolicyOnlinePb MujocoXarm7AdmittancePusht_T3 \
    --demo_name AdmittancePushTOnline_trainseed${train_seed}_rollseed42_T3 \
    --checkpoint "$online_checkpoint" \
    --wp4_checkpoint "$wp4_checkpoint" \
    --initial_object_id 0 \
    --online_pb_lr 2.5e-2 \
    --wrench_loss_weight 0.01 \
    --world_idx_list {370..379} \
    --seed 42 \
    --auto_exit \
    --max_duration 22 \
    --result_filename "$eval_dir/online_trainseed${train_seed}_rollseed42_T3.yaml" \
    --save_rollout \
    --no_plot \
    --no_render

  python robo_manip_baselines/bin/Rollout.py DiffusionPolicyOnlinePb MujocoXarm7AdmittancePusht_T4 \
    --demo_name AdmittancePushTOnline_trainseed${train_seed}_rollseed42_T4 \
    --checkpoint "$online_checkpoint" \
    --wp4_checkpoint "$wp4_checkpoint" \
    --initial_object_id 0 \
    --online_pb_lr 2.5e-2 \
    --wrench_loss_weight 0.01 \
    --world_idx_list {470..479} \
    --seed 42 \
    --auto_exit \
    --max_duration 22 \
    --result_filename "$eval_dir/online_trainseed${train_seed}_rollseed42_T4.yaml" \
    --save_rollout \
    --no_plot \
    --no_render

  # Constant-PB-trained Diffusion Policy with online PB adaptation.
  python robo_manip_baselines/bin/Rollout.py DiffusionPolicyOnlinePb MujocoXarm7AdmittancePusht_T0 \
    --demo_name AdmittancePushTConstant_trainseed${train_seed}_rollseed42_T0 \
    --checkpoint "$constant_checkpoint" \
    --wp4_checkpoint "$wp4_checkpoint" \
    --initial_object_id 0 \
    --online_pb_lr 2.5e-2 \
    --wrench_loss_weight 0.01 \
    --world_idx_list {70..79} \
    --seed 42 \
    --auto_exit \
    --max_duration 22 \
    --result_filename "$eval_dir/constant_trainseed${train_seed}_rollseed42_T0.yaml" \
    --save_rollout \
    --no_plot \
    --no_render

  python robo_manip_baselines/bin/Rollout.py DiffusionPolicyOnlinePb MujocoXarm7AdmittancePusht_T1 \
    --demo_name AdmittancePushTConstant_trainseed${train_seed}_rollseed42_T1 \
    --checkpoint "$constant_checkpoint" \
    --wp4_checkpoint "$wp4_checkpoint" \
    --initial_object_id 0 \
    --online_pb_lr 2.5e-2 \
    --wrench_loss_weight 0.01 \
    --world_idx_list {170..179} \
    --seed 42 \
    --auto_exit \
    --max_duration 22 \
    --result_filename "$eval_dir/constant_trainseed${train_seed}_rollseed42_T1.yaml" \
    --save_rollout \
    --no_plot \
    --no_render

  python robo_manip_baselines/bin/Rollout.py DiffusionPolicyOnlinePb MujocoXarm7AdmittancePusht_T2 \
    --demo_name AdmittancePushTConstant_trainseed${train_seed}_rollseed42_T2 \
    --checkpoint "$constant_checkpoint" \
    --wp4_checkpoint "$wp4_checkpoint" \
    --initial_object_id 0 \
    --online_pb_lr 2.5e-2 \
    --wrench_loss_weight 0.01 \
    --world_idx_list {270..279} \
    --seed 42 \
    --auto_exit \
    --max_duration 22 \
    --result_filename "$eval_dir/constant_trainseed${train_seed}_rollseed42_T2.yaml" \
    --save_rollout \
    --no_plot \
    --no_render

  python robo_manip_baselines/bin/Rollout.py DiffusionPolicyOnlinePb MujocoXarm7AdmittancePusht_T3 \
    --demo_name AdmittancePushTConstant_trainseed${train_seed}_rollseed42_T3 \
    --checkpoint "$constant_checkpoint" \
    --wp4_checkpoint "$wp4_checkpoint" \
    --initial_object_id 0 \
    --online_pb_lr 2.5e-2 \
    --wrench_loss_weight 0.01 \
    --world_idx_list {370..379} \
    --seed 42 \
    --auto_exit \
    --max_duration 22 \
    --result_filename "$eval_dir/constant_trainseed${train_seed}_rollseed42_T3.yaml" \
    --save_rollout \
    --no_plot \
    --no_render

  python robo_manip_baselines/bin/Rollout.py DiffusionPolicyOnlinePb MujocoXarm7AdmittancePusht_T4 \
    --demo_name AdmittancePushTConstant_trainseed${train_seed}_rollseed42_T4 \
    --checkpoint "$constant_checkpoint" \
    --wp4_checkpoint "$wp4_checkpoint" \
    --initial_object_id 0 \
    --online_pb_lr 2.5e-2 \
    --wrench_loss_weight 0.01 \
    --world_idx_list {470..479} \
    --seed 42 \
    --auto_exit \
    --max_duration 22 \
    --result_filename "$eval_dir/constant_trainseed${train_seed}_rollseed42_T4.yaml" \
    --save_rollout \
    --no_plot \
    --no_render
done

find robo_manip_baselines/dataset \
  -mindepth 1 \
  -maxdepth 1 \
  -type d \
  \( \
    -name "RolloutDiffusionPolicy_AdmittancePushTBaseline_trainseed*_rollseed42_T*_20*" \
    -o -name "RolloutDiffusionPolicyOnlinePb_AdmittancePushTOnline_trainseed*_rollseed42_T*_20*" \
    -o -name "RolloutDiffusionPolicyOnlinePb_AdmittancePushTConstant_trainseed*_rollseed42_T*_20*" \
  \) \
  -newer "$rollout_start_marker" \
  -exec mv -- {} "$rmb_dir" \;

