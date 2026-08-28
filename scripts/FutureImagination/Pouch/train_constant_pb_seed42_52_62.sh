#!/usr/bin/env bash
set -e

# Run after prepare_datasets.sh. Each episode uses its object's learned PB.
dataset_dir=robo_manip_baselines/dataset/0827_DatasetPouch_RightVAE9_Wp4JointPos_ConstantPB/training
checkpoint_prefix=robo_manip_baselines/checkpoint/DiffusionPolicy/DpImageRight/0827_DatasetPouch_RightVAE9_Wp4JointPos_EefPose_ConstantPB

python robo_manip_baselines/bin/Train.py DiffusionPolicy \
  --dataset_dir "$dataset_dir" \
  --checkpoint_dir "${checkpoint_prefix}_seed42" \
  --camera_names right \
  --state_keys measured_eef_pose material_property \
  --action_keys command_eef_pose \
  --train_ratio 1.0 \
  --val_ratio 0.01

python robo_manip_baselines/bin/Train.py DiffusionPolicy \
  --dataset_dir "$dataset_dir" \
  --checkpoint_dir "${checkpoint_prefix}_seed52" \
  --camera_names right \
  --state_keys measured_eef_pose material_property \
  --action_keys command_eef_pose \
  --train_ratio 1.0 \
  --val_ratio 0.01 \
  --seed 52

python robo_manip_baselines/bin/Train.py DiffusionPolicy \
  --dataset_dir "$dataset_dir" \
  --checkpoint_dir "${checkpoint_prefix}_seed62" \
  --camera_names right \
  --state_keys measured_eef_pose material_property \
  --action_keys command_eef_pose \
  --train_ratio 1.0 \
  --val_ratio 0.01 \
  --seed 62
