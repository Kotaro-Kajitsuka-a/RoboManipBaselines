#!/bin/sh
#PBS -q rt_HG
#PBS -l select=1
#PBS -l walltime=30:00:00
#PBS -P gag51454

source ~/miniconda3/bin/activate
conda activate robo_diff_FI

cd ~/projects/ForceImagination/RoboManipBaselines/

set -e

# Train with each episode's beta=2 WP4 constant PB using seed 52.
dataset_dir=robo_manip_baselines/dataset/0827_DatasetPouch_RightVAE9Beta2_Wp4JointPos_ConstantPB/training
checkpoint_dir=robo_manip_baselines/checkpoint/DiffusionPolicy/DpImageRight/0827_DatasetPouch_RightVAE9Beta2_Wp4JointPos_EefPose_ConstantPB_seed52

python robo_manip_baselines/bin/Train.py DiffusionPolicy \
  --dataset_dir "$dataset_dir" \
  --checkpoint_dir "$checkpoint_dir" \
  --camera_names right \
  --state_keys measured_eef_pose measured_gripper_pos material_property \
  --action_keys command_eef_pose command_gripper_pos \
  --train_ratio 1.0 \
  --val_ratio 0.01 \
  --seed 52
