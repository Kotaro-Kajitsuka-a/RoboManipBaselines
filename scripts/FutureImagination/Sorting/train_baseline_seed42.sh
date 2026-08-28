#!/bin/sh
#PBS -q rt_HG
#PBS -l select=1
#PBS -l walltime=30:00:00
#PBS -P gag51454

source ~/miniconda3/bin/activate
conda activate robo_diff_FI

cd ~/projects/ForceImagination/RoboManipBaselines/

set -e

# Train front-image Diffusion Policy without PB input using seed 42.
dataset_dir=robo_manip_baselines/dataset/0826_DatasetSorting/training
checkpoint_dir=robo_manip_baselines/checkpoint/DiffusionPolicy/DpImageFront/0826_DatasetSorting_EefPose_Baseline_seed42

python robo_manip_baselines/bin/Train.py DiffusionPolicy \
  --dataset_dir "$dataset_dir" \
  --checkpoint_dir "$checkpoint_dir" \
  --camera_names front \
  --state_keys measured_eef_pose \
  --action_keys command_eef_pose \
  --train_ratio 1.0 \
  --val_ratio 0.01
