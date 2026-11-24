#!/bin/sh
#PBS -q rt_HG
#PBS -l select=1
#PBS -l walltime=30:00:00
#PBS -P gag51454

source ~/miniconda3/bin/activate
conda activate Flex

cd ~/projects/FlexiblePressureSensor/RoboManipBaselines/


set -e

# Train right-image Diffusion Policy with wrench and without PB using seed 42.
dataset_dir=robo_manip_baselines/dataset/Datasets/1124_cookie/
checkpoint_dir=robo_manip_baselines/checkpoint/DiffusionPolicy/1124_cookie_right_hand_pressure_seed42/



python robo_manip_baselines/bin/Train.py DiffusionPolicy \
  --dataset_dir "$dataset_dir" \
  --checkpoint_dir "$checkpoint_dir" \
  --camera_names right hand\
  --state_keys measured_eef_pose measured_eef_wrench main_pressure_sensor \
  --action_keys command_eef_pose \
  --train_ratio 1.0 \
  --val_ratio 0.01
