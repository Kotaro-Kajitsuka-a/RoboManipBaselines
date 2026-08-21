#!/usr/bin/env bash
set -e

# Run from the repository root after training the seed-42 Gaussian Online-PB DP.
eval_timestamp=$(date +%Y%m%d_%H%M%S)
eval_dir="robo_manip_baselines/dataset/tests/FutureImagination/DatasetMujocoUR5eLiftingi/DP_eef_pose_gaussian_online_pb_m16_seed42_eval_${eval_timestamp}"
rmb_dir="$eval_dir/rmb"
checkpoint=robo_manip_baselines/checkpoint/DiffusionPolicy/DatasetMujocoUR5eLiftingi_EefPose_GaussianOnline_m9_beta10_seed42_sleepblocked/policy_last.ckpt
wp4_checkpoint=/home/kotaro/my_projects/robotics/RoboManipBaselines/robo_manip_baselines/checkpoint/WrenchPredictor4/DatasetMujocoUR5eLiftingi_EefPose/policy_best.ckpt
mkdir -p "$rmb_dir"
rollout_start_marker="$eval_dir/.rollout_start"
touch "$rollout_start_marker"

python -m robo_manip_baselines.bin.Rollout \
  DiffusionPolicyOnlinePb MujocoUR5eLiftingi_I0 \
  --demo_name MujocoUR5eLiftingi_I0_eef_pose_gaussian_online_pb_m16_trainseed42 \
  --checkpoint "$checkpoint" \
  --wp4_checkpoint "$wp4_checkpoint" \
  --online_pb_update_type gaussian_belief \
  --online_pb_initial_std 0.25 \
  --online_pb_num_points 16 \
  --online_pb_beta 10 \
  --wrench_loss_weight 0.0 \
  --seed 42 \
  --world_idx_list {70..79} \
  --auto_exit \
  --max_duration 17 \
  --save_rollout \
  --result_filename "$eval_dir/gaussian_online_pb_m16_trainseed42_I0.yaml" \
  --no_plot \
  --no_render

python -m robo_manip_baselines.bin.Rollout \
  DiffusionPolicyOnlinePb MujocoUR5eLiftingi_I1 \
  --demo_name MujocoUR5eLiftingi_I1_eef_pose_gaussian_online_pb_m16_trainseed42 \
  --checkpoint "$checkpoint" \
  --wp4_checkpoint "$wp4_checkpoint" \
  --online_pb_update_type gaussian_belief \
  --online_pb_initial_std 0.25 \
  --online_pb_num_points 16 \
  --online_pb_beta 10 \
  --wrench_loss_weight 0.0 \
  --seed 42 \
  --world_idx_list {170..179} \
  --auto_exit \
  --max_duration 17 \
  --save_rollout \
  --result_filename "$eval_dir/gaussian_online_pb_m16_trainseed42_I1.yaml" \
  --no_plot \
  --no_render

python -m robo_manip_baselines.bin.Rollout \
  DiffusionPolicyOnlinePb MujocoUR5eLiftingi_I2 \
  --demo_name MujocoUR5eLiftingi_I2_eef_pose_gaussian_online_pb_m16_trainseed42 \
  --checkpoint "$checkpoint" \
  --wp4_checkpoint "$wp4_checkpoint" \
  --online_pb_update_type gaussian_belief \
  --online_pb_initial_std 0.25 \
  --online_pb_num_points 16 \
  --online_pb_beta 10 \
  --wrench_loss_weight 0.0 \
  --seed 42 \
  --world_idx_list {270..279} \
  --auto_exit \
  --max_duration 17 \
  --save_rollout \
  --result_filename "$eval_dir/gaussian_online_pb_m16_trainseed42_I2.yaml" \
  --no_plot \
  --no_render

python -m robo_manip_baselines.bin.Rollout \
  DiffusionPolicyOnlinePb MujocoUR5eLiftingi_I4 \
  --demo_name MujocoUR5eLiftingi_I4_eef_pose_gaussian_online_pb_m16_trainseed42 \
  --checkpoint "$checkpoint" \
  --wp4_checkpoint "$wp4_checkpoint" \
  --online_pb_update_type gaussian_belief \
  --online_pb_initial_std 0.25 \
  --online_pb_num_points 16 \
  --online_pb_beta 10 \
  --wrench_loss_weight 0.0 \
  --seed 42 \
  --world_idx_list {470..479} \
  --auto_exit \
  --max_duration 17 \
  --save_rollout \
  --result_filename "$eval_dir/gaussian_online_pb_m16_trainseed42_I4.yaml" \
  --no_plot \
  --no_render

python -m robo_manip_baselines.bin.Rollout \
  DiffusionPolicyOnlinePb MujocoUR5eLiftingi_I5 \
  --demo_name MujocoUR5eLiftingi_I5_eef_pose_gaussian_online_pb_m16_trainseed42 \
  --checkpoint "$checkpoint" \
  --wp4_checkpoint "$wp4_checkpoint" \
  --online_pb_update_type gaussian_belief \
  --online_pb_initial_std 0.25 \
  --online_pb_num_points 16 \
  --online_pb_beta 10 \
  --wrench_loss_weight 0.0 \
  --seed 42 \
  --world_idx_list {570..579} \
  --auto_exit \
  --max_duration 17 \
  --save_rollout \
  --result_filename "$eval_dir/gaussian_online_pb_m16_trainseed42_I5.yaml" \
  --no_plot \
  --no_render

python -m robo_manip_baselines.bin.Rollout \
  DiffusionPolicyOnlinePb MujocoUR5eLiftingi_I6 \
  --demo_name MujocoUR5eLiftingi_I6_eef_pose_gaussian_online_pb_m16_trainseed42 \
  --checkpoint "$checkpoint" \
  --wp4_checkpoint "$wp4_checkpoint" \
  --online_pb_update_type gaussian_belief \
  --online_pb_initial_std 0.25 \
  --online_pb_num_points 16 \
  --online_pb_beta 10 \
  --wrench_loss_weight 0.0 \
  --seed 42 \
  --world_idx_list {670..679} \
  --auto_exit \
  --max_duration 17 \
  --save_rollout \
  --result_filename "$eval_dir/gaussian_online_pb_m16_trainseed42_I6.yaml" \
  --no_plot \
  --no_render

python -m robo_manip_baselines.bin.Rollout \
  DiffusionPolicyOnlinePb MujocoUR5eLiftingi_I7 \
  --demo_name MujocoUR5eLiftingi_I7_eef_pose_gaussian_online_pb_m16_trainseed42 \
  --checkpoint "$checkpoint" \
  --wp4_checkpoint "$wp4_checkpoint" \
  --online_pb_update_type gaussian_belief \
  --online_pb_initial_std 0.25 \
  --online_pb_num_points 16 \
  --online_pb_beta 10 \
  --wrench_loss_weight 0.0 \
  --seed 42 \
  --world_idx_list {770..779} \
  --auto_exit \
  --max_duration 17 \
  --save_rollout \
  --result_filename "$eval_dir/gaussian_online_pb_m16_trainseed42_I7.yaml" \
  --no_plot \
  --no_render

for object_id in 0 1 2 4 5 6 7; do
  object_dir="$rmb_dir/WrenchPredObject${object_id}"
  mkdir -p "$object_dir"
  find robo_manip_baselines/dataset \
    -mindepth 1 \
    -maxdepth 1 \
    -type d \
    -name "RolloutDiffusionPolicyOnlinePb_MujocoUR5eLiftingi_I${object_id}_eef_pose_gaussian_online_pb_m16_trainseed42_20*" \
    -newer "$rollout_start_marker" \
    -exec mv -- {} "$object_dir" \;
done

python -m robo_manip_baselines.misc.futureimagination.AnalyzeLiftingSuccess \
  "$rmb_dir" \
  --output_dir "$eval_dir" \
  --output_prefix eef_pose_gaussian_online_pb_m16_trainseed42 \
  --expected_episode_count 70

echo "evaluation directory: $eval_dir"
