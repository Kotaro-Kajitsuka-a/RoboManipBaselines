#!/usr/bin/env bash
set -e

if [ "$#" -ne 1 ]; then
  echo "Usage: bash $0 <run_name>"
  exit 1
fi

run_name=$1
source_dataset=robo_manip_baselines/dataset/DatasetMujocoUR5eLiftingi
dataset_root=robo_manip_baselines/dataset/DatasetMujocoUR5eLiftingi_LeftScratchVAE9_Fresh_${run_name}
train_dataset=$dataset_root/training
validation_dataset=$dataset_root/validation
online_dataset=$dataset_root/training_online_pb_lr6e3
vae_output_dir=robo_manip_baselines/checkpoint/ImageVAE/DatasetMujocoUR5eLiftingi_left_9_Fresh_${run_name}
wp4_checkpoint_dir=robo_manip_baselines/checkpoint/WrenchPredictor4/DatasetMujocoUR5eLiftingi_left_image_vae_9_joint_pos_Fresh_${run_name}
wp4_checkpoint=$wp4_checkpoint_dir/policy_best.ckpt

if [ -e "$dataset_root" ] || [ -e "$vae_output_dir" ] || [ -e "$wp4_checkpoint_dir" ]; then
  echo "Fresh output already exists for run_name=$run_name"
  echo "Choose a new run name; existing data and checkpoints are never reused."
  exit 1
fi

mkdir -p "$dataset_root"
cp -aL "$source_dataset/training" "$train_dataset"
cp -aL "$source_dataset/validation" "$validation_dataset"

python robo_manip_baselines/misc/futureimagination/TrainImageVAE.py \
  --train_dataset_dir "$train_dataset" \
  --validation_dataset_dir "$validation_dataset" \
  --output_dir "$vae_output_dir" \
  --camera_name left \
  --latent_dim 9

python robo_manip_baselines/misc/futureimagination/AddImageVAEFeature.py \
  "$train_dataset" \
  --checkpoint "$vae_output_dir/final_model" \
  --camera_name left \
  --overwrite

python robo_manip_baselines/misc/futureimagination/AddImageVAEFeature.py \
  "$validation_dataset" \
  --checkpoint "$vae_output_dir/final_model" \
  --camera_name left \
  --overwrite

python robo_manip_baselines/bin/Train.py WrenchPredictor4 \
  --dataset_dir "$train_dataset" \
  --val_dataset_dir "$validation_dataset" \
  --checkpoint_dir "$wp4_checkpoint_dir" \
  --camera_names \
  --state_keys measured_joint_pos \
  --action_keys command_joint_pos \
  --image_feature_key image_vae_left_9 \
  --wrench_source_key measured_eef_wrench \
  --pb_dim 1 \
  --wrench_loss_weight 0.1 \
  --num_epochs 500 \
  --seed 42

python robo_manip_baselines/policy/wrench_predictor4/EvalWrenchPredictor4ImageFeatureSweepDir.py \
  "$wp4_checkpoint" \
  "$validation_dataset" \
  --material_object_ids 0 1 2

cp -aL "$train_dataset" "$online_dataset"

python robo_manip_baselines/policy/wrench_predictor4_online/AddOnlinePbToDataset.py \
  "$online_dataset" \
  0 \
  --checkpoint "$wp4_checkpoint" \
  --lr 6e-3 \
  --wrench_loss_weight 0.0 \
  --seed 42 \
  --overwrite

python robo_manip_baselines/misc/futureimagination/PlotOnlinePbDataset.py \
  "$online_dataset/WrenchPredObject0" \
  --output "$online_dataset/WrenchPredObject0_online_pb_trajectories.png"

python robo_manip_baselines/misc/futureimagination/PlotOnlinePbDataset.py \
  "$online_dataset/WrenchPredObject1" \
  --output "$online_dataset/WrenchPredObject1_online_pb_trajectories.png"

python robo_manip_baselines/misc/futureimagination/PlotOnlinePbDataset.py \
  "$online_dataset/WrenchPredObject2" \
  --output "$online_dataset/WrenchPredObject2_online_pb_trajectories.png"

sha256sum "$vae_output_dir/final_model/model.pt" "$wp4_checkpoint"

echo "$online_dataset/WrenchPredObject0_online_pb_trajectories.png"
echo "$online_dataset/WrenchPredObject1_online_pb_trajectories.png"
echo "$online_dataset/WrenchPredObject2_online_pb_trajectories.png"
