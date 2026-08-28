#!/usr/bin/env bash
set -e

# Train the baseline joint-conditioned CVAE and matched joint-position WP4.
train_dataset_dir=robo_manip_baselines/dataset/0827_DatasetPouch/training
validation_dataset_dir=robo_manip_baselines/dataset/0827_DatasetPouch/validation
cvae_output_dir=robo_manip_baselines/checkpoint/ImageJointConditionedCVAE/0827_DatasetPouch_right_9_joint_pos
wp4_checkpoint_dir=robo_manip_baselines/checkpoint/WrenchPredictor4/0827_DatasetPouch_right_image_cvae_9_joint_pos
image_feature_key=image_cvae_right_joint_pos_9

python robo_manip_baselines/misc/futureimagination/TrainImageJointConditionedCVAE.py \
  --train_dataset_dir "$train_dataset_dir" \
  --validation_dataset_dir "$validation_dataset_dir" \
  --output_dir "$cvae_output_dir" \
  --camera_name right \
  --condition_keys measured_joint_pos \
  --latent_dim 9 \
  --num_epochs 100

python robo_manip_baselines/misc/futureimagination/AddImageJointConditionedCVAEFeature.py \
  "$train_dataset_dir" \
  --checkpoint "$cvae_output_dir/final_model" \
  --camera_name right \
  --output_key "$image_feature_key" \
  --overwrite

python robo_manip_baselines/misc/futureimagination/AddImageJointConditionedCVAEFeature.py \
  "$validation_dataset_dir" \
  --checkpoint "$cvae_output_dir/final_model" \
  --camera_name right \
  --output_key "$image_feature_key" \
  --overwrite

python robo_manip_baselines/bin/Train.py WrenchPredictor4 \
  --dataset_dir "$train_dataset_dir" \
  --val_dataset_dir "$validation_dataset_dir" \
  --checkpoint_dir "$wp4_checkpoint_dir" \
  --camera_names \
  --state_keys measured_joint_pos \
  --action_keys command_joint_pos \
  --image_feature_key "$image_feature_key" \
  --wrench_source_key measured_eef_wrench \
  --pb_dim 1 \
  --wrench_loss_weight 0.0 \
  --num_epochs 500

python robo_manip_baselines/policy/wrench_predictor4/EvalWrenchPredictor4ImageFeatureSweepDir.py \
  "$wp4_checkpoint_dir/policy_best.ckpt" \
  "$validation_dataset_dir" \
  --material_object_ids 0 1 2

python robo_manip_baselines/misc/futureimagination/MakeJointConditionedCVAEReconstructionVideos.py \
  "$train_dataset_dir" \
  "$cvae_output_dir/final_model" \
  --camera_name right

python robo_manip_baselines/misc/futureimagination/MakeJointConditionedCVAEReconstructionVideos.py \
  "$validation_dataset_dir" \
  "$cvae_output_dir/final_model" \
  --camera_name right
