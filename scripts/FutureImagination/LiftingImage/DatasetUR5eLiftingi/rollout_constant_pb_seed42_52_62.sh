#!/usr/bin/env bash
set -e

eval_timestamp=$(date +%Y%m%d_%H%M%S)
eval_dir="robo_manip_baselines/dataset/tests/FutureImagination/DatasetMujocoUR5eLiftingi_Image/DP_constant_pb_eval_${eval_timestamp}"
rmb_dir="$eval_dir/rmb"
mkdir -p "$rmb_dir"
rollout_start_marker="$eval_dir/.rollout_start"
touch "$rollout_start_marker"

pids=()
for train_seed in 42 52 62; do
  (

  python robo_manip_baselines/bin/Rollout.py \
    DiffusionPolicyOnlinePb MujocoUR5eLiftingi_I0 \
    --demo_name "MujocoUR5eLiftingi_I0_image_constant_pb_trainseed${train_seed}" \
    --checkpoint "robo_manip_baselines/checkpoint/DiffusionPolicy/DatasetMujocoUR5eLiftingi_Image_ConstantPB_seed${train_seed}/policy_last.ckpt" \
    --wp4_checkpoint robo_manip_baselines/checkpoint/WrenchPredictor4/DatasetMujocoUR5eLiftingi_hand_image_vae_9/policy_best.ckpt \
    --image_vae_checkpoint robo_manip_baselines/checkpoint/ImageVAE/DatasetMujocoUR5eLiftingi_hand_9/final_model \
    --online_pb_lr 6e-3 \
    --wrench_loss_weight 0.0 \
    --seed 42 \
    --world_idx_list {70..79} \
    --auto_exit \
    --max_duration 17 \
    --save_rollout \
    --result_filename "$eval_dir/constant_pb_trainseed${train_seed}_I0.yaml" \
    --no_plot \
    --no_render

  python robo_manip_baselines/bin/Rollout.py \
    DiffusionPolicyOnlinePb MujocoUR5eLiftingi_I1 \
    --demo_name "MujocoUR5eLiftingi_I1_image_constant_pb_trainseed${train_seed}" \
    --checkpoint "robo_manip_baselines/checkpoint/DiffusionPolicy/DatasetMujocoUR5eLiftingi_Image_ConstantPB_seed${train_seed}/policy_last.ckpt" \
    --wp4_checkpoint robo_manip_baselines/checkpoint/WrenchPredictor4/DatasetMujocoUR5eLiftingi_hand_image_vae_9/policy_best.ckpt \
    --image_vae_checkpoint robo_manip_baselines/checkpoint/ImageVAE/DatasetMujocoUR5eLiftingi_hand_9/final_model \
    --online_pb_lr 6e-3 \
    --wrench_loss_weight 0.0 \
    --seed 42 \
    --world_idx_list {170..179} \
    --auto_exit \
    --max_duration 17 \
    --save_rollout \
    --result_filename "$eval_dir/constant_pb_trainseed${train_seed}_I1.yaml" \
    --no_plot \
    --no_render

  python robo_manip_baselines/bin/Rollout.py \
    DiffusionPolicyOnlinePb MujocoUR5eLiftingi_I2 \
    --demo_name "MujocoUR5eLiftingi_I2_image_constant_pb_trainseed${train_seed}" \
    --checkpoint "robo_manip_baselines/checkpoint/DiffusionPolicy/DatasetMujocoUR5eLiftingi_Image_ConstantPB_seed${train_seed}/policy_last.ckpt" \
    --wp4_checkpoint robo_manip_baselines/checkpoint/WrenchPredictor4/DatasetMujocoUR5eLiftingi_hand_image_vae_9/policy_best.ckpt \
    --image_vae_checkpoint robo_manip_baselines/checkpoint/ImageVAE/DatasetMujocoUR5eLiftingi_hand_9/final_model \
    --online_pb_lr 6e-3 \
    --wrench_loss_weight 0.0 \
    --seed 42 \
    --world_idx_list {270..279} \
    --auto_exit \
    --max_duration 17 \
    --save_rollout \
    --result_filename "$eval_dir/constant_pb_trainseed${train_seed}_I2.yaml" \
    --no_plot \
    --no_render

  python robo_manip_baselines/bin/Rollout.py \
    DiffusionPolicyOnlinePb MujocoUR5eLiftingi_I4 \
    --demo_name "MujocoUR5eLiftingi_I4_image_constant_pb_trainseed${train_seed}" \
    --checkpoint "robo_manip_baselines/checkpoint/DiffusionPolicy/DatasetMujocoUR5eLiftingi_Image_ConstantPB_seed${train_seed}/policy_last.ckpt" \
    --wp4_checkpoint robo_manip_baselines/checkpoint/WrenchPredictor4/DatasetMujocoUR5eLiftingi_hand_image_vae_9/policy_best.ckpt \
    --image_vae_checkpoint robo_manip_baselines/checkpoint/ImageVAE/DatasetMujocoUR5eLiftingi_hand_9/final_model \
    --online_pb_lr 6e-3 \
    --wrench_loss_weight 0.0 \
    --seed 42 \
    --world_idx_list {470..479} \
    --auto_exit \
    --max_duration 17 \
    --save_rollout \
    --result_filename "$eval_dir/constant_pb_trainseed${train_seed}_I4.yaml" \
    --no_plot \
    --no_render

  python robo_manip_baselines/bin/Rollout.py \
    DiffusionPolicyOnlinePb MujocoUR5eLiftingi_I5 \
    --demo_name "MujocoUR5eLiftingi_I5_image_constant_pb_trainseed${train_seed}" \
    --checkpoint "robo_manip_baselines/checkpoint/DiffusionPolicy/DatasetMujocoUR5eLiftingi_Image_ConstantPB_seed${train_seed}/policy_last.ckpt" \
    --wp4_checkpoint robo_manip_baselines/checkpoint/WrenchPredictor4/DatasetMujocoUR5eLiftingi_hand_image_vae_9/policy_best.ckpt \
    --image_vae_checkpoint robo_manip_baselines/checkpoint/ImageVAE/DatasetMujocoUR5eLiftingi_hand_9/final_model \
    --online_pb_lr 6e-3 \
    --wrench_loss_weight 0.0 \
    --seed 42 \
    --world_idx_list {570..579} \
    --auto_exit \
    --max_duration 17 \
    --save_rollout \
    --result_filename "$eval_dir/constant_pb_trainseed${train_seed}_I5.yaml" \
    --no_plot \
    --no_render

  python robo_manip_baselines/bin/Rollout.py \
    DiffusionPolicyOnlinePb MujocoUR5eLiftingi_I6 \
    --demo_name "MujocoUR5eLiftingi_I6_image_constant_pb_trainseed${train_seed}" \
    --checkpoint "robo_manip_baselines/checkpoint/DiffusionPolicy/DatasetMujocoUR5eLiftingi_Image_ConstantPB_seed${train_seed}/policy_last.ckpt" \
    --wp4_checkpoint robo_manip_baselines/checkpoint/WrenchPredictor4/DatasetMujocoUR5eLiftingi_hand_image_vae_9/policy_best.ckpt \
    --image_vae_checkpoint robo_manip_baselines/checkpoint/ImageVAE/DatasetMujocoUR5eLiftingi_hand_9/final_model \
    --online_pb_lr 6e-3 \
    --wrench_loss_weight 0.0 \
    --seed 42 \
    --world_idx_list {670..679} \
    --auto_exit \
    --max_duration 17 \
    --save_rollout \
    --result_filename "$eval_dir/constant_pb_trainseed${train_seed}_I6.yaml" \
    --no_plot \
    --no_render

  python robo_manip_baselines/bin/Rollout.py \
    DiffusionPolicyOnlinePb MujocoUR5eLiftingi_I7 \
    --demo_name "MujocoUR5eLiftingi_I7_image_constant_pb_trainseed${train_seed}" \
    --checkpoint "robo_manip_baselines/checkpoint/DiffusionPolicy/DatasetMujocoUR5eLiftingi_Image_ConstantPB_seed${train_seed}/policy_last.ckpt" \
    --wp4_checkpoint robo_manip_baselines/checkpoint/WrenchPredictor4/DatasetMujocoUR5eLiftingi_hand_image_vae_9/policy_best.ckpt \
    --image_vae_checkpoint robo_manip_baselines/checkpoint/ImageVAE/DatasetMujocoUR5eLiftingi_hand_9/final_model \
    --online_pb_lr 6e-3 \
    --wrench_loss_weight 0.0 \
    --seed 42 \
    --world_idx_list {770..779} \
    --auto_exit \
    --max_duration 17 \
    --save_rollout \
    --result_filename "$eval_dir/constant_pb_trainseed${train_seed}_I7.yaml" \
    --no_plot \
    --no_render

  for object_id in 0 1 2 4 5 6 7; do
    object_dir="$rmb_dir/WrenchPredObject${object_id}"
    mkdir -p "$object_dir"
    find robo_manip_baselines/dataset \
      -mindepth 1 \
      -maxdepth 1 \
      -type d \
      -name "RolloutDiffusionPolicyOnlinePb_MujocoUR5eLiftingi_I${object_id}_image_constant_pb_trainseed${train_seed}_20*" \
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
  --output_prefix constant_pb_training_seed_42_52_62 \
  --expected_episode_count 210
