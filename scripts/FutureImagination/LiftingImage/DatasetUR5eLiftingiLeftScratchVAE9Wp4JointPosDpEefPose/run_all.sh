#!/usr/bin/env bash
set -e

# Reuse the existing Left ImageVAE 9D and JointPos WP4. Recompute this
# experiment's training PB labels, train the two EEF-pose Diffusion Policy
# conditions, and evaluate them.
experiment_base=scripts/FutureImagination/LiftingImage/DatasetUR5eLiftingiLeftScratchVAE9Wp4JointPosDpEefPose

python -c "import torch; assert torch.cuda.is_available(), 'CUDA GPU is required'"
bash "$experiment_base/prepare_datasets.sh"
test -f robo_manip_baselines/checkpoint/WrenchPredictor4/DatasetMujocoUR5eLiftingi_left_image_vae_9_joint_pos/policy_best.ckpt
test -d robo_manip_baselines/checkpoint/ImageVAE/DatasetMujocoUR5eLiftingi_left_9/final_model

bash "$experiment_base/train_DP_online_pb_recomputed_seed42_52_62.sh"
bash "$experiment_base/train_constant_pb_recomputed_seed42_52_62.sh"
bash "$experiment_base/rollout_DP_online_pb_recomputed_seed42_52_62.sh"
bash "$experiment_base/rollout_constant_pb_recomputed_seed42_52_62.sh"
