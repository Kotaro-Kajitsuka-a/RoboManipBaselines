#!/usr/bin/env bash
set -e

# Run from the repository root in the activated virtual environment.
# Use the original WP4 that supplied the existing constant-PB DP's training PBs.
# This is the source path recorded in the constant rollouts; edit it if relocated.
fixed_pb_checkpoint=/home/kotaro/my_projects/robot/RoboManipBaselines/robo_manip_baselines/checkpoint/WrenchPredictor4/DatasetPushingLine_front_image_vae_9_joint_pos/policy_best.ckpt

source_dataset_dir=robo_manip_baselines/dataset/DatasetPushingLine
rollout_dataset_dir=robo_manip_baselines/dataset/tests/FutureImaginationReal/PushingLine_2/constant
validation_rollout_dataset_dir=robo_manip_baselines/dataset/tests/FutureImaginationReal/PushingLine_2/baseline/seed62
combined_dataset_dir=robo_manip_baselines/dataset/DatasetPushingLine_PostTrain
train_dataset_dir="$combined_dataset_dir/training"
validation_dataset_dir="$combined_dataset_dir/validation"
vae_output_dir=robo_manip_baselines/checkpoint/ImageVAE/DatasetPushingLine_PostTrain_front_9
wp4_checkpoint_dir=robo_manip_baselines/checkpoint/WrenchPredictor4/DatasetPushingLine_PostTrain_front_image_vae_9_joint_pos

if [ ! -f "$fixed_pb_checkpoint" ]; then
  echo "WP4 checkpoint not found: $fixed_pb_checkpoint" >&2
  exit 1
fi
# Stop if the combined dataset already exists, avoiding stale or overwritten data.
mkdir "$combined_dataset_dir"
mkdir -p \
  "$train_dataset_dir/rollout/seed42" \
  "$train_dataset_dir/rollout/seed52" \
  "$train_dataset_dir/rollout/seed62" \
  "$validation_dataset_dir/rollout/baseline/seed62"

# Preserve the demo split and add all constant-PB rollout seeds to training.
cp -r "$source_dataset_dir/training" "$train_dataset_dir/demo"
cp -r "$source_dataset_dir/validation" "$validation_dataset_dir/demo"
# Match only Object0 through Object4, excluding Object-1 and fractional IDs.
cp -r "$rollout_dataset_dir/seed42"/WrenchPredObject[0-4] "$train_dataset_dir/rollout/seed42/"
cp -r "$rollout_dataset_dir/seed52"/WrenchPredObject[0-4] "$train_dataset_dir/rollout/seed52/"
cp -r "$rollout_dataset_dir/seed62"/WrenchPredObject[0-4] "$train_dataset_dir/rollout/seed62/"
# Baseline seed62 must first be organized into WrenchPredObject0 through 4 folders.
cp -r "$validation_rollout_dataset_dir"/WrenchPredObject[0-4] "$validation_dataset_dir/rollout/baseline/seed62/"

# Train a new VAE, then encode every copied episode with that same VAE.
python robo_manip_baselines/misc/futureimagination/TrainImageVAE.py \
  --train_dataset_dir "$train_dataset_dir" \
  --validation_dataset_dir "$validation_dataset_dir" \
  --output_dir "$vae_output_dir" \
  --camera_name front \
  --latent_dim 9

python robo_manip_baselines/misc/futureimagination/AddImageVAEFeature.py \
  "$train_dataset_dir" \
  --checkpoint "$vae_output_dir/final_model" \
  --camera_name front \
  --overwrite

python robo_manip_baselines/misc/futureimagination/AddImageVAEFeature.py \
  "$validation_dataset_dir" \
  --checkpoint "$vae_output_dir/final_model" \
  --camera_name front \
  --overwrite

# Train WP4 from scratch, loading and freezing only the original PB table.
python robo_manip_baselines/bin/Train.py WrenchPredictor4 \
  --dataset_dir "$train_dataset_dir" \
  --val_dataset_dir "$validation_dataset_dir" \
  --checkpoint_dir "$wp4_checkpoint_dir" \
  --fixed_pb_checkpoint "$fixed_pb_checkpoint" \
  --camera_names \
  --state_keys measured_joint_pos \
  --action_keys command_joint_pos \
  --image_feature_key image_vae_front_9 \
  --wrench_source_key measured_eef_wrench \
  --pb_dim 1 \
  --num_epochs 500

python robo_manip_baselines/policy/wrench_predictor4/EvalWrenchPredictor4ImageFeatureSweepDir.py \
  "$wp4_checkpoint_dir/policy_best.ckpt" \
  "$validation_dataset_dir" \
  --material_object_ids 0 1 2 3 4
