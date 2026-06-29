#!/bin/bash

set -eu

cd "$(dirname "$0")/.."

CHECKPOINT_DIR=${CHECKPOINT_DIR:-robo_manip_baselines/checkpoint/DiffusionWorldModel/pinch_marker_delta_dwm_h16}
TRAIN_DIR=${TRAIN_DIR:-robo_manip_baselines/dataset/ピンチテスト_marker/training}
VAL_DIR=${VAL_DIR:-robo_manip_baselines/dataset/ピンチテスト_marker/validation}

uv run python robo_manip_baselines/bin/Train.py DiffusionWorldModel \
  --dataset_dir "${TRAIN_DIR}" \
  --checkpoint_dir "${CHECKPOINT_DIR}" \
  --state_keys measured_eef_pose measured_gripper_joint_pos \
  --action_keys command_eef_pose command_gripper_joint_pos \
  --image_feature_key front_apriltag_pose_xy_axis \
  --image_feature_target_mode delta_from_last_obs \
  --wrench_source_key measured_eef_wrench \
  --scheduler ddim \
  --horizon 16 \
  --n_obs_steps 2 \
  --pb_dim 9 \
  --num_epochs 1000 \
  --batch_size 64 \
  --train_ratio 0.99 \
  --val_ratio 0.01

uv run python robo_manip_baselines/policy/diffusion_world_model/EvalDiffusionWorldModelMarkerSweepDir.py \
  "${CHECKPOINT_DIR}" \
  "${VAL_DIR}" \
  --max_material_object_id 2 \
  --checkpoint_names policy_best.ckpt policy_last.ckpt \
  --output_suffix _delta \
  --no_plot
