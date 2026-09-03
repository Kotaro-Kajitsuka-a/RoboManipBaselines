#!/usr/bin/env bash
set -e

# Run from the repository root after train_constant_pb_seed42_52_62.sh.
# Evaluate the constant-PB-trained DP while identifying PB online.
eval_timestamp=$(date +%Y%m%d_%H%M%S)
eval_dir="robo_manip_baselines/dataset/tests/FutureImagination/AdmittancePushtAB/DP_constant_pb_eval_${eval_timestamp}"
rmb_dir="$eval_dir/rmb"
mkdir -p "$rmb_dir"
rollout_start_marker="$eval_dir/.rollout_start"
touch "$rollout_start_marker"

wp4_checkpoint=robo_manip_baselines/checkpoint/WrenchPredictor4/AdmittancePushtAB/policy_best.ckpt

# Run 10 episodes for T0-T3 under every training seed.
for train_seed in 42 52 62; do

  python robo_manip_baselines/bin/Rollout.py \
    DiffusionPolicyOnlinePb MujocoXarm7AdmittancePusht_T0 \
    --demo_name "AdmittancePushtABConstant_trainseed${train_seed}_rollseed42_T0" \
    --checkpoint "robo_manip_baselines/checkpoint/DiffusionPolicy/AdmittancePushtAB/ConstantPB_seed${train_seed}/policy_last.ckpt" \
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

  python robo_manip_baselines/bin/Rollout.py \
    DiffusionPolicyOnlinePb MujocoXarm7AdmittancePusht_T1 \
    --demo_name "AdmittancePushtABConstant_trainseed${train_seed}_rollseed42_T1" \
    --checkpoint "robo_manip_baselines/checkpoint/DiffusionPolicy/AdmittancePushtAB/ConstantPB_seed${train_seed}/policy_last.ckpt" \
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

  python robo_manip_baselines/bin/Rollout.py \
    DiffusionPolicyOnlinePb MujocoXarm7AdmittancePusht_T2 \
    --demo_name "AdmittancePushtABConstant_trainseed${train_seed}_rollseed42_T2" \
    --checkpoint "robo_manip_baselines/checkpoint/DiffusionPolicy/AdmittancePushtAB/ConstantPB_seed${train_seed}/policy_last.ckpt" \
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

  python robo_manip_baselines/bin/Rollout.py \
    DiffusionPolicyOnlinePb MujocoXarm7AdmittancePusht_T3 \
    --demo_name "AdmittancePushtABConstant_trainseed${train_seed}_rollseed42_T3" \
    --checkpoint "robo_manip_baselines/checkpoint/DiffusionPolicy/AdmittancePushtAB/ConstantPB_seed${train_seed}/policy_last.ckpt" \
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

  for object_id in 0 1 2 3; do
    object_dir="$rmb_dir/WrenchPredObject${object_id}"
    mkdir -p "$object_dir"
    find robo_manip_baselines/dataset \
      -mindepth 1 \
      -maxdepth 1 \
      -type d \
      -name "RolloutDiffusionPolicyOnlinePb_AdmittancePushtABConstant_trainseed${train_seed}_rollseed42_T${object_id}_20*" \
      -newer "$rollout_start_marker" \
      -exec mv -- {} "$object_dir" \;
  done
done
