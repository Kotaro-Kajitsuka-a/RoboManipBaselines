#!/usr/bin/env bash
set -e

# Run from the repository root after train_constant_pb_seed42_52_62.sh.
# The DP was trained with oracle constant PB, but evaluation identifies PB
# online in exactly the same way as the proposed method.
eval_timestamp=$(date +%Y%m%d_%H%M%S)
eval_dir="robo_manip_baselines/dataset/tests/FutureImagination/DatasetMujocoUR5eLiftingi/DP_joint_pos_constant_pb_eval_${eval_timestamp}"
rmb_dir="$eval_dir/rmb"
wp4_checkpoint=robo_manip_baselines/checkpoint/WrenchPredictor4/DatasetMujocoUR5eLiftingi_JointPos/policy_best.ckpt
mkdir -p "$rmb_dir"
rollout_start_marker="$eval_dir/.rollout_start"
touch "$rollout_start_marker"

# Run 10 episodes for each object under every training seed.
pids=()
for train_seed in 42 52 62; do
  (
  seed_dir="$rmb_dir/seed${train_seed}"
  checkpoint="robo_manip_baselines/checkpoint/DiffusionPolicy/DatasetMujocoUR5eLiftingi_JointPos_ConstantPB_seed${train_seed}/policy_last.ckpt"
  mkdir -p "$seed_dir"

  python robo_manip_baselines/bin/Rollout.py \
    DiffusionPolicyOnlinePb MujocoUR5eLiftingi_I0 \
    --demo_name "MujocoUR5eLiftingi_I0_joint_pos_constant_pb_trainseed${train_seed}" \
    --checkpoint "$checkpoint" \
    --wp4_checkpoint "$wp4_checkpoint" \
    --wrench_loss_weight 0.0 \
    --seed 42 \
    --world_idx_list {70..79} \
    --auto_exit \
    --max_duration 20 \
    --save_rollout \
    --result_filename "$eval_dir/constant_pb_trainseed${train_seed}_I0.yaml" \
    --no_plot \
    --no_render

  python robo_manip_baselines/bin/Rollout.py \
    DiffusionPolicyOnlinePb MujocoUR5eLiftingi_I1 \
    --demo_name "MujocoUR5eLiftingi_I1_joint_pos_constant_pb_trainseed${train_seed}" \
    --checkpoint "$checkpoint" \
    --wp4_checkpoint "$wp4_checkpoint" \
    --wrench_loss_weight 0.0 \
    --seed 42 \
    --world_idx_list {170..179} \
    --auto_exit \
    --max_duration 20 \
    --save_rollout \
    --result_filename "$eval_dir/constant_pb_trainseed${train_seed}_I1.yaml" \
    --no_plot \
    --no_render

  python robo_manip_baselines/bin/Rollout.py \
    DiffusionPolicyOnlinePb MujocoUR5eLiftingi_I2 \
    --demo_name "MujocoUR5eLiftingi_I2_joint_pos_constant_pb_trainseed${train_seed}" \
    --checkpoint "$checkpoint" \
    --wp4_checkpoint "$wp4_checkpoint" \
    --wrench_loss_weight 0.0 \
    --seed 42 \
    --world_idx_list {270..279} \
    --auto_exit \
    --max_duration 20 \
    --save_rollout \
    --result_filename "$eval_dir/constant_pb_trainseed${train_seed}_I2.yaml" \
    --no_plot \
    --no_render

  python robo_manip_baselines/bin/Rollout.py \
    DiffusionPolicyOnlinePb MujocoUR5eLiftingi_I4 \
    --demo_name "MujocoUR5eLiftingi_I4_joint_pos_constant_pb_trainseed${train_seed}" \
    --checkpoint "$checkpoint" \
    --wp4_checkpoint "$wp4_checkpoint" \
    --wrench_loss_weight 0.0 \
    --seed 42 \
    --world_idx_list {470..479} \
    --auto_exit \
    --max_duration 20 \
    --save_rollout \
    --result_filename "$eval_dir/constant_pb_trainseed${train_seed}_I4.yaml" \
    --no_plot \
    --no_render

  python robo_manip_baselines/bin/Rollout.py \
    DiffusionPolicyOnlinePb MujocoUR5eLiftingi_I5 \
    --demo_name "MujocoUR5eLiftingi_I5_joint_pos_constant_pb_trainseed${train_seed}" \
    --checkpoint "$checkpoint" \
    --wp4_checkpoint "$wp4_checkpoint" \
    --wrench_loss_weight 0.0 \
    --seed 42 \
    --world_idx_list {570..579} \
    --auto_exit \
    --max_duration 20 \
    --save_rollout \
    --result_filename "$eval_dir/constant_pb_trainseed${train_seed}_I5.yaml" \
    --no_plot \
    --no_render

  python robo_manip_baselines/bin/Rollout.py \
    DiffusionPolicyOnlinePb MujocoUR5eLiftingi_I6 \
    --demo_name "MujocoUR5eLiftingi_I6_joint_pos_constant_pb_trainseed${train_seed}" \
    --checkpoint "$checkpoint" \
    --wp4_checkpoint "$wp4_checkpoint" \
    --wrench_loss_weight 0.0 \
    --seed 42 \
    --world_idx_list {670..679} \
    --auto_exit \
    --max_duration 20 \
    --save_rollout \
    --result_filename "$eval_dir/constant_pb_trainseed${train_seed}_I6.yaml" \
    --no_plot \
    --no_render

  python robo_manip_baselines/bin/Rollout.py \
    DiffusionPolicyOnlinePb MujocoUR5eLiftingi_I7 \
    --demo_name "MujocoUR5eLiftingi_I7_joint_pos_constant_pb_trainseed${train_seed}" \
    --checkpoint "$checkpoint" \
    --wp4_checkpoint "$wp4_checkpoint" \
    --wrench_loss_weight 0.0 \
    --seed 42 \
    --world_idx_list {770..779} \
    --auto_exit \
    --max_duration 20 \
    --save_rollout \
    --result_filename "$eval_dir/constant_pb_trainseed${train_seed}_I7.yaml" \
    --no_plot \
    --no_render

  find robo_manip_baselines/dataset \
    -mindepth 1 \
    -maxdepth 1 \
    -type d \
    -name "RolloutDiffusionPolicyOnlinePb_MujocoUR5eLiftingi_I*_joint_pos_constant_pb_trainseed${train_seed}_20*" \
    -newer "$rollout_start_marker" \
    -exec mv -- {} "$seed_dir" \;
  ) &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "$pid"
done

# Evaluate the 210 saved RMB episodes with the same final-state criterion used
# for baseline and proposed: lift at least 10 cm and tilt less than 7.5 degrees.
python robo_manip_baselines/misc/futureimagination/AnalyzeLiftingSuccess.py \
  "$rmb_dir" \
  --output_dir "$eval_dir" \
  --output_prefix joint_pos_constant_pb_training_seed_42_52_62 \
  --expected_episode_count 210
