#!/usr/bin/env bash
set -e

# Run from the repository root with the virtual environment activated.
dataset_dir=robo_manip_baselines/dataset/DatasetMujocoXarm7AdmittancePusht
validation_dataset_dir=robo_manip_baselines/dataset/DatasetMujocoXarm7AdmittancePushtValidation/validation
vae_output_dir=robo_manip_baselines/checkpoint/ImageVAE/DatasetMujocoXarm7AdmittancePusht_hand_9_20260814
wp4_checkpoint_dir=robo_manip_baselines/checkpoint/WrenchPredictor4/DatasetMujocoXarm7AdmittancePusht_hand_image_vae_9_20260814

python robo_manip_baselines/misc/futureimagination/TrainImageVAE.py \
  --train_dataset_dir "$dataset_dir/training" \
  --validation_dataset_dir "$validation_dataset_dir" \
  --output_dir "$vae_output_dir" \
  --camera_name hand \
  --latent_dim 9

python robo_manip_baselines/misc/futureimagination/AddImageVAEFeature.py \
  "$dataset_dir/training" \
  --checkpoint "$vae_output_dir/final_model" \
  --camera_name hand

python robo_manip_baselines/bin/Train.py WrenchPredictor4 \
  --dataset_dir "$dataset_dir/training" \
  --checkpoint_dir "$wp4_checkpoint_dir" \
  --camera_names \
  --state_keys measured_eef_pose \
  --action_keys command_eef_pose \
  --image_feature_key image_vae_hand_9 \
  --wrench_source_key measured_eef_wrench \
  --pb_dim 2 \
  --wrench_loss_weight 0.1 \
  --num_epochs 500 \
  --seed 42
