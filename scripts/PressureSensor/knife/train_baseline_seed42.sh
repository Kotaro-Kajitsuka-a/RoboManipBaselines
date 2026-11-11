#!/bin/sh
#PBS -q rt_HG
#PBS -l select=1
#PBS -l walltime=30:00:00
#PBS -P gag51454

source ~/miniconda3/bin/activate
conda activate robo_diff_FI

cd ~/projects/ForceImagination/RoboManipBaselines/

set -e

# Train right-image Diffusion Policy with wrench and without PB using seed 42.
dataset_dir=robo_manip_baselines/dataset/Datasets/1111_knife/
checkpoint_dir=robo_manip_baselines/checkpoint/DiffusionPolicy/1111_knife/

python robo_manip_baselines/bin/Train.py DiffusionPolicy \
  --dataset_dir "$dataset_dir" \
  --checkpoint_dir "$checkpoint_dir" \
  --camera_names right \
  --state_keys measured_eef_pose measured_eef_wrench \
  --action_keys command_eef_pose \
  --train_ratio 1.0 \
  --val_ratio 0.01
