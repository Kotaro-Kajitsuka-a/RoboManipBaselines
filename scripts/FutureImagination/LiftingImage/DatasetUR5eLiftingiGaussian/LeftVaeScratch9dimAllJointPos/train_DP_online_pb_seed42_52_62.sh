#!/usr/bin/env bash
set -e

dataset_dir=robo_manip_baselines/dataset/DatasetMujocoUR5eLiftingi_LeftVaeScratch9dim_JointPos_GaussianOnlinePB_beta10_m16_20260821_v1
checkpoint_prefix=robo_manip_baselines/checkpoint/DiffusionPolicy/DatasetMujocoUR5eLiftingi_LeftVaeScratch9dim_JointPos_GaussianOnlinePB_beta10_m16_20260821_v1

test -d "$dataset_dir"
test ! -e "${checkpoint_prefix}_seed42"
test ! -e "${checkpoint_prefix}_seed52"
test ! -e "${checkpoint_prefix}_seed62"

python robo_manip_baselines/bin/Train.py DiffusionPolicy \
  --dataset_dir "$dataset_dir" \
  --checkpoint_dir "${checkpoint_prefix}_seed42" \
  --camera_names \
  --state_keys measured_joint_pos measured_tblock_pose material_property online_pb_std \
  --action_keys command_joint_pos \
  --backbone cnn \
  --scheduler ddim \
  --skip 3 \
  --batch_size 64 \
  --num_workers 2 \
  --num_epochs 500 \
  --lr 1e-4 \
  --train_ratio 1.0 \
  --val_ratio 0.01 \
  --seed 42

python robo_manip_baselines/bin/Train.py DiffusionPolicy \
  --dataset_dir "$dataset_dir" \
  --checkpoint_dir "${checkpoint_prefix}_seed52" \
  --camera_names \
  --state_keys measured_joint_pos measured_tblock_pose material_property online_pb_std \
  --action_keys command_joint_pos \
  --backbone cnn \
  --scheduler ddim \
  --skip 3 \
  --batch_size 64 \
  --num_workers 2 \
  --num_epochs 500 \
  --lr 1e-4 \
  --train_ratio 1.0 \
  --val_ratio 0.01 \
  --seed 52

python robo_manip_baselines/bin/Train.py DiffusionPolicy \
  --dataset_dir "$dataset_dir" \
  --checkpoint_dir "${checkpoint_prefix}_seed62" \
  --camera_names \
  --state_keys measured_joint_pos measured_tblock_pose material_property online_pb_std \
  --action_keys command_joint_pos \
  --backbone cnn \
  --scheduler ddim \
  --skip 3 \
  --batch_size 64 \
  --num_workers 2 \
  --num_epochs 500 \
  --lr 1e-4 \
  --train_ratio 1.0 \
  --val_ratio 0.01 \
  --seed 62
