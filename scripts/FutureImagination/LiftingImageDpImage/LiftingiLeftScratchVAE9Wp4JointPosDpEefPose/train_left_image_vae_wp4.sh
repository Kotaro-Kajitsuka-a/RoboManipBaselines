#!/usr/bin/env bash
set -e

# Train a left-camera VAE from scratch, add its 9D features, and train WP4
# with joint-position state/action inputs on the legacy lifting dataset.
train_dataset_dir=robo_manip_baselines/dataset/LiftingAB_B_only
validation_dataset_dir=robo_manip_baselines/dataset/LiftingAB_B_only_Validation
vae_output_dir=robo_manip_baselines/checkpoint/ImageVAE/LiftingAB_B_only_left_9
wp4_checkpoint_dir=robo_manip_baselines/checkpoint/WrenchPredictor4/LiftingAB_B_only_left_image_vae_9_joint_pos

python robo_manip_baselines/misc/futureimagination/TrainImageVAE.py \
  --train_dataset_dir "$train_dataset_dir" \
  --validation_dataset_dir "$validation_dataset_dir" \
  --output_dir "$vae_output_dir" \
  --camera_name left \
  --latent_dim 9

python robo_manip_baselines/misc/futureimagination/AddImageVAEFeature.py \
  "$train_dataset_dir" \
  --checkpoint "$vae_output_dir/final_model" \
  --camera_name left \
  --overwrite

python robo_manip_baselines/misc/futureimagination/AddImageVAEFeature.py \
  "$validation_dataset_dir" \
  --checkpoint "$vae_output_dir/final_model" \
  --camera_name left \
  --overwrite

python robo_manip_baselines/bin/Train.py WrenchPredictor4 \
  --dataset_dir "$train_dataset_dir" \
  --val_dataset_dir "$validation_dataset_dir" \
  --checkpoint_dir "$wp4_checkpoint_dir" \
  --camera_names \
  --state_keys measured_joint_pos \
  --action_keys command_joint_pos \
  --image_feature_key image_vae_left_9 \
  --wrench_source_key measured_eef_wrench \
  --pb_dim 1 \
  --wrench_loss_weight 0.1 \
  --num_epochs 500

python robo_manip_baselines/policy/wrench_predictor4/EvalWrenchPredictor4ImageFeatureSweepDir.py \
  "$wp4_checkpoint_dir/policy_best.ckpt" \
  "$validation_dataset_dir" \
  --material_object_ids 0 1 2
