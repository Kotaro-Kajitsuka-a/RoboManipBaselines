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
check_path robo_manip_baselines/checkpoint/WrenchPredictor4/DatasetMujocoUR5eLiftingi_left_image_vae_9_joint_pos/policy_best.ckpt
check_path robo_manip_baselines/checkpoint/ImageVAE/DatasetMujocoUR5eLiftingi_left_9/final_model/model_config.json
check_path robo_manip_baselines/checkpoint/ImageVAE/DatasetMujocoUR5eLiftingi_left_9/final_model/model.pt
check_path robo_manip_baselines/checkpoint/ImageVAE/DatasetMujocoUR5eLiftingi_left_9/final_model/encoder.pkl
check_path robo_manip_baselines/checkpoint/ImageVAE/DatasetMujocoUR5eLiftingi_left_9/final_model/decoder.pkl
check_path robo_manip_baselines/checkpoint/DiffusionPolicy/DpImageLeft/DatasetMujocoUR5eLiftingi_LeftScratchVAE9_Wp4JointPos_DpEefPose_ConstantPB_seed42/policy_last.ckpt
check_path robo_manip_baselines/checkpoint/DiffusionPolicy/DpImageLeft/DatasetMujocoUR5eLiftingi_LeftScratchVAE9_Wp4JointPos_DpEefPose_ConstantPB_seed52/policy_last.ckpt
check_path robo_manip_baselines/checkpoint/DiffusionPolicy/DpImageLeft/DatasetMujocoUR5eLiftingi_LeftScratchVAE9_Wp4JointPos_DpEefPose_ConstantPB_seed62/policy_last.ckpt

if [ "$missing" -ne 0 ]; then
  exit 1
fi

echo "All required files for the image-based SGD experiment are present."
