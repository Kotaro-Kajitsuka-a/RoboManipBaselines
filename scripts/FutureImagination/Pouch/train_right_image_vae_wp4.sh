#!/usr/bin/env bash
set -e

# Run from the repository root with the virtual environment activated.
train_dataset_dir=robo_manip_baselines/dataset/0827_DatasetPouch/training
validation_dataset_dir=robo_manip_baselines/dataset/0827_DatasetPouch/validation
vae_output_dir=robo_manip_baselines/checkpoint/ImageVAE/0827_DatasetPouch_right_9
wp4_checkpoint_dir=robo_manip_baselines/checkpoint/WrenchPredictor4/0827_DatasetPouch_image_vae_9_joint_pos

python robo_manip_baselines/misc/futureimagination/TrainImageVAE.py \
  --train_dataset_dir "$train_dataset_dir" \
  --validation_dataset_dir "$validation_dataset_dir" \
  --output_dir "$vae_output_dir" \
  --camera_name right \
  --latent_dim 9

python robo_manip_baselines/misc/futureimagination/AddImageVAEFeature.py \
  "$train_dataset_dir" \
  --checkpoint "$vae_output_dir/final_model" \
  --camera_name right \
  --overwrite

python robo_manip_baselines/misc/futureimagination/AddImageVAEFeature.py \
  "$validation_dataset_dir" \
  --checkpoint "$vae_output_dir/final_model" \
  --camera_name right \
  --overwrite

python robo_manip_baselines/bin/Train.py WrenchPredictor4 \
  --dataset_dir "$train_dataset_dir" \
  --val_dataset_dir "$validation_dataset_dir" \
  --checkpoint_dir "$wp4_checkpoint_dir" \
  --camera_names \
  --state_keys measured_joint_pos \
  --action_keys command_joint_pos \
  --image_feature_key image_vae_right_9 \
  --wrench_source_key measured_eef_wrench \
  --pb_dim 1 \
  --wrench_loss_weight 0.0 \
  --num_epochs 500

python robo_manip_baselines/policy/wrench_predictor4/EvalWrenchPredictor4ImageFeatureSweepDir.py \
  "$wp4_checkpoint_dir/policy_best.ckpt" \
  "$validation_dataset_dir" \
  --material_object_ids 0 1 2

python robo_manip_baselines/misc/futureimagination/MakeReconstructionVideos.py \
  "$train_dataset_dir" \
  "$vae_output_dir/final_model" \
  --camera_name right

python robo_manip_baselines/misc/futureimagination/MakeReconstructionVideos.py \
  "$validation_dataset_dir" \
  "$vae_output_dir/final_model" \
  --camera_name right
