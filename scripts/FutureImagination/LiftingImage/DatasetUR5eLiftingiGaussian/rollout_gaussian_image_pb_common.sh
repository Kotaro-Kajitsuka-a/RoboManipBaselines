#!/usr/bin/env bash
set -e

# This file is sourced by the camera/state-specific rollout scripts.
: "${eval_tag:?}"
: "${demo_tag:?}"
: "${method_name:?}"
: "${checkpoint_prefix:?}"
: "${wp4_checkpoint:?}"
: "${image_vae_checkpoint:?}"
: "${image_vae_camera_name:?}"

eval_timestamp=$(date +%Y%m%d_%H%M%S)
eval_dir="robo_manip_baselines/dataset/tests/FutureImagination/DatasetMujocoUR5eLiftingiGaussian/${eval_tag}_eval_${eval_timestamp}"
rmb_dir="$eval_dir/rmb"
mkdir -p "$rmb_dir"
rollout_start_marker="$eval_dir/.rollout_start"
touch "$rollout_start_marker"

pids=()
for train_seed in 42 52 62; do
  (
  checkpoint="${checkpoint_prefix}_seed${train_seed}/policy_last.ckpt"

  for object_id in 0 1 2 4 5 6 7; do
    case "$object_id" in
      0) world_idx_list=({70..79}) ;;
      1) world_idx_list=({170..179}) ;;
      2) world_idx_list=({270..279}) ;;
      4) world_idx_list=({470..479}) ;;
      5) world_idx_list=({570..579}) ;;
      6) world_idx_list=({670..679}) ;;
      7) world_idx_list=({770..779}) ;;
    esac

    python robo_manip_baselines/bin/Rollout.py \
      DiffusionPolicyOnlinePb "MujocoUR5eLiftingi_I${object_id}" \
      --demo_name "MujocoUR5eLiftingi_I${object_id}_${demo_tag}_trainseed${train_seed}" \
      --checkpoint "$checkpoint" \
      --wp4_checkpoint "$wp4_checkpoint" \
      --image_vae_checkpoint "$image_vae_checkpoint" \
      --image_vae_camera_name "$image_vae_camera_name" \
      --online_pb_update_type gaussian_belief \
      --online_pb_initial_std 0.25 \
      --online_pb_num_points 16 \
      --online_pb_beta 10 \
      --wrench_loss_weight 0.0 \
      --seed 42 \
      --world_idx_list "${world_idx_list[@]}" \
      --auto_exit \
      --max_duration 17 \
      --save_rollout \
      --result_filename "$eval_dir/${method_name}_trainseed${train_seed}_I${object_id}.yaml" \
      --no_plot \
      --no_render
  done

  for object_id in 0 1 2 4 5 6 7; do
    object_dir="$rmb_dir/WrenchPredObject${object_id}"
    mkdir -p "$object_dir"
    find robo_manip_baselines/dataset \
      -mindepth 1 \
      -maxdepth 1 \
      -type d \
      -name "RolloutDiffusionPolicyOnlinePb_MujocoUR5eLiftingi_I${object_id}_${demo_tag}_trainseed${train_seed}_20*" \
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
  --output_prefix "${method_name}_training_seed_42_52_62" \
  --expected_episode_count 210

echo "evaluation directory: $eval_dir"
