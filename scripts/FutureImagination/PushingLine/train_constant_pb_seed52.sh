#!/bin/sh
#PBS -q rt_HG
#PBS -l select=1
#PBS -l walltime=30:00:00
#PBS -P gag51454

source ~/miniconda3/bin/activate
conda activate robo_diff_FI

cd ~/projects/ForceImagination/RoboManipBaselines/

set -e

# Train with wrench and each episode's learned constant PB using seed 52.
dataset_dir=robo_manip_baselines/dataset/DatasetPushingLine_FrontVAE9_Wp4JointPos_ConstantPB/training
checkpoint_dir=robo_manip_baselines/checkpoint/DiffusionPolicy/DpImageFront/DatasetPushingLine_FrontVAE9_Wp4JointPos_EefPoseWrench_ConstantPB_seed52

python robo_manip_baselines/bin/Train.py DiffusionPolicy \
  --dataset_dir "$dataset_dir" \
  --checkpoint_dir "$checkpoint_dir" \
  --camera_names front \
  --state_keys measured_eef_pose measured_eef_wrench material_property \
  --action_keys command_eef_pose \
  --train_ratio 1.0 \
  --val_ratio 0.01 \
  --seed 52
