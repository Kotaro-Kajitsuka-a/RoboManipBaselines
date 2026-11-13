#!/usr/bin/env bash
set -e

eval_timestamp=$(date +%Y%m%d_%H%M%S)
eval_dir="robo_manip_baselines/dataset/tests/FutureImagination/DatasetMujocoUR5eLiftingi_LeftScratchVAE9_Wp4JointPos_DpEefPose/DP_constant_pb_oracle_pb_fixed_eval_${eval_timestamp}"
rmb_dir="$eval_dir/rmb"
wp4_checkpoint=robo_manip_baselines/checkpoint/WrenchPredictor4/DatasetMujocoUR5eLiftingi_left_image_vae_9_joint_pos/policy_best.ckpt
image_vae_checkpoint=robo_manip_baselines/checkpoint/ImageVAE/DatasetMujocoUR5eLiftingi_left_9/final_model
mkdir -p "$rmb_dir"
rollout_start_marker="$eval_dir/.rollout_start"
touch "$rollout_start_marker"

pids=()
for train_seed in 42 52 62; do
  (
  dp_checkpoint="robo_manip_baselines/checkpoint/DiffusionPolicy/DatasetMujocoUR5eLiftingi_LeftScratchVAE9_Wp4JointPos_DpEefPose_ConstantPB_seed${train_seed}/policy_last.ckpt"

  python robo_manip_baselines/bin/Rollout.py \
    DiffusionPolicyOnlinePb MujocoUR5eLiftingi_I0 \
    --demo_name "MujocoUR5eLiftingi_I0_left_scratch_vae9_wp4_joint_dp_eef_constant_pb_oracle_pb_fixed_trainseed${train_seed}" \
    --checkpoint "$dp_checkpoint" \
    --wp4_checkpoint "$wp4_checkpoint" \
    --image_vae_checkpoint "$image_vae_checkpoint" \
    --image_vae_camera_name left \
    --online_pb_lr 6e-3 \
    --wrench_loss_weight 0.0 \
    --initial_object_id 0 \
    --fixed_pb \
    --seed 42 \
    --world_idx_list {70..79} \
    --auto_exit \
    --max_duration 17 \
    --save_rollout \
    --result_filename "$eval_dir/oracle_pb_fixed_trainseed${train_seed}_I0.yaml" \
    --no_plot \
    --no_render

  python robo_manip_baselines/bin/Rollout.py \
    DiffusionPolicyOnlinePb MujocoUR5eLiftingi_I1 \
    --demo_name "MujocoUR5eLiftingi_I1_left_scratch_vae9_wp4_joint_dp_eef_constant_pb_oracle_pb_fixed_trainseed${train_seed}" \
    --checkpoint "$dp_checkpoint" \
    --wp4_checkpoint "$wp4_checkpoint" \
    --image_vae_checkpoint "$image_vae_checkpoint" \
    --image_vae_camera_name left \
    --online_pb_lr 6e-3 \
    --wrench_loss_weight 0.0 \
    --initial_object_id 1 \
    --fixed_pb \
    --seed 42 \
    --world_idx_list {170..179} \
    --auto_exit \
    --max_duration 17 \
    --save_rollout \
    --result_filename "$eval_dir/oracle_pb_fixed_trainseed${train_seed}_I1.yaml" \
    --no_plot \
    --no_render

  python robo_manip_baselines/bin/Rollout.py \
    DiffusionPolicyOnlinePb MujocoUR5eLiftingi_I2 \
    --demo_name "MujocoUR5eLiftingi_I2_left_scratch_vae9_wp4_joint_dp_eef_constant_pb_oracle_pb_fixed_trainseed${train_seed}" \
    --checkpoint "$dp_checkpoint" \
    --wp4_checkpoint "$wp4_checkpoint" \
    --image_vae_checkpoint "$image_vae_checkpoint" \
    --image_vae_camera_name left \
    --online_pb_lr 6e-3 \
    --wrench_loss_weight 0.0 \
    --initial_object_id 2 \
    --fixed_pb \
    --seed 42 \
    --world_idx_list {270..279} \
    --auto_exit \
    --max_duration 17 \
    --save_rollout \
    --result_filename "$eval_dir/oracle_pb_fixed_trainseed${train_seed}_I2.yaml" \
    --no_plot \
    --no_render

  for object_id in 0 1 2; do
    object_dir="$rmb_dir/WrenchPredObject${object_id}"
    mkdir -p "$object_dir"
    find robo_manip_baselines/dataset \
      -mindepth 1 \
      -maxdepth 1 \
      -type d \
      -name "RolloutDiffusionPolicyOnlinePb_MujocoUR5eLiftingi_I${object_id}_left_scratch_vae9_wp4_joint_dp_eef_constant_pb_oracle_pb_fixed_trainseed${train_seed}_20*" \
      -newer "$rollout_start_marker" \
      -exec mv -- {} "$object_dir" \;
  done
  ) &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "$pid"
done

python robo_manip_baselines/misc/futureimagination/AnalyzeLiftingSuccess.py \
  "$rmb_dir" \
  --output_dir "$eval_dir" \
  --output_prefix oracle_pb_fixed_training_seed_42_52_62 \
  --expected_episode_count 90
