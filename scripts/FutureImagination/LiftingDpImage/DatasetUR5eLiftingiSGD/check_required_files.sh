#!/usr/bin/env bash

# Run from the repository root before run_all.sh.
missing=0

check_path() {
  if [ ! -e "$1" ]; then
    echo "MISSING: $1"
    missing=1
  fi
}

check_path robo_manip_baselines/dataset/DatasetMujocoUR5eLiftingi/training
check_path robo_manip_baselines/checkpoint/WrenchPredictor4/DatasetMujocoUR5eLiftingi_EefPose/policy_best.ckpt
check_path robo_manip_baselines/checkpoint/DiffusionPolicy/DpImageLeft/DatasetMujocoUR5eLiftingi_EefPose_ConstantPB_seed42/policy_last.ckpt
check_path robo_manip_baselines/checkpoint/DiffusionPolicy/DpImageLeft/DatasetMujocoUR5eLiftingi_EefPose_ConstantPB_seed52/policy_last.ckpt
check_path robo_manip_baselines/checkpoint/DiffusionPolicy/DpImageLeft/DatasetMujocoUR5eLiftingi_EefPose_ConstantPB_seed62/policy_last.ckpt

if [ "$missing" -ne 0 ]; then
  exit 1
fi

echo "All required files for DatasetUR5eLiftingiSGD are present."
