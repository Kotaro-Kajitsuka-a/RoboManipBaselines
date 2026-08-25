#!/usr/bin/env bash
set -e

# Run from the repository root with the virtual environment activated.
train_dataset_dir=robo_manip_baselines/dataset/0825_DatasetSorting_20260825_134504/training
validation_dataset_dir=robo_manip_baselines/dataset/0825_DatasetSorting_20260825_134504/validation
sd3_ae_output_dir=robo_manip_baselines/checkpoint/SD3LatentAE/0825_DatasetSorting_20260825_front_12
wp4_checkpoint_dir=robo_manip_baselines/checkpoint/WrenchPredictor4/0825_DatasetSorting_20260825_front_sd3_ae_12_joint_pos

python robo_manip_baselines/misc/futureimagination/AddImageFeature.py \
  "$train_dataset_dir" \
  --camera_name front \
  --output_key sd3_vae_front \
  --overwrite

python robo_manip_baselines/misc/futureimagination/AddImageFeature.py \
  "$validation_dataset_dir" \
  --camera_name front \
  --output_key sd3_vae_front \
  --overwrite

python robo_manip_baselines/misc/futureimagination/TrainSD3LatentAE.py \
  --train_dataset_dir "$train_dataset_dir" \
  --validation_dataset_dir "$validation_dataset_dir" \
  --output_dir "$sd3_ae_output_dir" \
  --image_feature_key sd3_vae_front \
  --latent_dim 12

python robo_manip_baselines/misc/futureimagination/AddSD3LatentAEFeature.py \
  "$train_dataset_dir" \
  --checkpoint "$sd3_ae_output_dir/final_model" \
  --source_key sd3_vae_front \
  --output_key sd3_vae_front_ae_12 \
  --overwrite

python robo_manip_baselines/misc/futureimagination/AddSD3LatentAEFeature.py \
  "$validation_dataset_dir" \
  --checkpoint "$sd3_ae_output_dir/final_model" \
  --source_key sd3_vae_front \
  --output_key sd3_vae_front_ae_12 \
  --overwrite

python robo_manip_baselines/bin/Train.py WrenchPredictor4 \
  --dataset_dir "$train_dataset_dir" \
  --val_dataset_dir "$validation_dataset_dir" \
  --checkpoint_dir "$wp4_checkpoint_dir" \
  --camera_names \
  --state_keys measured_joint_pos \
  --action_keys command_joint_pos \
  --image_feature_key sd3_vae_front_ae_12 \
  --wrench_source_key measured_eef_wrench \
  --pb_dim 1 \
  --wrench_loss_weight 0.0 \
  --num_epochs 500

python robo_manip_baselines/policy/wrench_predictor4/EvalWrenchPredictor4ImageFeatureSweepDir.py \
  "$wp4_checkpoint_dir/policy_best.ckpt" \
  "$validation_dataset_dir" \
  --material_object_ids 0 1 2

python robo_manip_baselines/misc/futureimagination/MakeSD3LatentAEReconstructionVideos.py \
  "$validation_dataset_dir" \
  "$sd3_ae_output_dir/final_model" \
  --camera_name front
