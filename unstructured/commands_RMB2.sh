#Teleopration with vive
python ./robo_manip_baselines/bin/Teleop.py MujocoXarm7DualAdmittanceBox --input_device vive --input_device_config  ./robo_manip_baselines/teleop/configs/ViveDual.yaml --world_idx_list 0 1


#ただのteleopration
python robo_manip_baselines/bin/Teleop.py MujocoXarm7AdmittancePusht

#replay_log
python robo_manip_baselines/bin/Teleop.py MujocoXarm7AdmittancePusht \
  --replay_log \
  robo_manip_baselines/dataset/MujocoXarm7Pusht_20260420_134319/MujocoXarm7Pusht_world20_002.rmb/ \
  --world_idx_list 10





# WrenchPredictor2 training (Admittance dataset)
python robo_manip_baselines/bin/Train.py WrenchPredictor2   \
  --dataset_dir robo_manip_baselines/dataset/DatasetAdmittancePushtNotPushingTable/training/same_color \
   --state_keys measured_eef_pose measured_eef_pose_rel    --action_keys command_eef_pose   --chunk_size 1 \
     --lr 3e-5 --num_epochs 100 --batch_size 32 --camera_name front right

python robo_manip_baselines/policy/wrench_predictor2/EvalWrenchPredictor.py \
  /path/to/checkpoint/policy_best.ckpt \
  robo_manip_baselines/dataset/DatasetAdmittancePushtNotPushingTable/validation

python ./bin/Train.py DiffusionPolicy \
  --camera_names front \
  --state_keys measured_eef_pose \
  --action_keys command_eef_pose \
  --dataset_dir ./dataset/DatasetColl/training_50 \
  --scheduler ddim --num_epochs 1000 \
  --train_ratio 1.0 --val_ratio 0.01

python ./robo_manip_baselines/bin/Rollout.py DiffusionPolicy MujocoXarm7AdmittancePusht \
  --checkpoint robo_manip_baselines/checkpoint/DiffusionPolicy/ \
  --wait_before_start --skip_draw 50000 --save_rollout --world_idx_repeat_count 10

# Run WrenchPredictor3 Coll training/evaluation locally without submitting a PBS job
bash scripts/train_WrenchPred_Coll.sh

7月30日
python robo_manip_baselines/bin/Train.py DiffusionPolicy \
    --dataset_dir robo_manip_baselines/dataset/MujocoUR5eLiftingi_I1_20260730_171251 \
    --camera_names \
    --state_keys \
      measured_eef_pose \
      measured_gripper_joint_pos \
      measured_tblock_pose \
    --action_keys \
      command_eef_pose \
      command_gripper_joint_pos \
    --scheduler ddim \
    --horizon 16 \
    --n_obs_steps 2 \
    --n_action_steps 8 \
    --batch_size 16 \
    --num_workers 2 \
    --num_epochs 500 \
    --train_ratio 1.0 --val_ratio 0.01

python robo_manip_baselines/bin/Rollout.py \
    DiffusionPolicy \
    MujocoUR5eLiftingi_I1 \
    --checkpoint /path/to/policy_best.ckpt \
    --world_idx_list {111..149} \
    --auto_exit \
    --max_duration 8 \
    --save_rollout \
    --no_plot



##Proposed Method#####################################################################
# Train WrenchPredictor4 on disjoint I0/I1/I2 world indices (absolute 9D pose, 1D PB, MLP-only)
uv run python robo_manip_baselines/bin/Train.py WrenchPredictor4 \
  --dataset_dir robo_manip_baselines/dataset/LiftingDisjointWorldIdx/training \
  --val_dataset_dir robo_manip_baselines/dataset/LiftingDisjointWorldIdx/validation \
  --checkpoint_dir robo_manip_baselines/checkpoint/WrenchPredictor4/LiftingAB_B_only/ \
  --camera_names --state_keys measured_eef_pose measured_gripper_joint_pos --action_keys command_eef_pose command_gripper_joint_pos \
  --image_feature_key measured_tblock_pose --wrench_source_key measured_eef_wrench --pb_dim 1 \
  --output_head mlp_only --wrench_loss_weight 0.1 --scheduler ddpm --horizon 16 --n_obs_steps 2 \
  --skip 3 --batch_size 64 --num_epochs 500 --lr 1e-4

# Evaluate policy_best.ckpt by sweeping the Object0/Object1/Object2 material PBs
python robo_manip_baselines/policy/wrench_predictor4/EvalWrenchPredictor4LiftingSweepDir.py \
  robo_manip_baselines/checkpoint/WrenchPredictor4/LiftingAB_B_only/ \
  robo_manip_baselines/dataset/LiftingAB_B_only_Validation \
  --checkpoint_names policy_best.ckpt \
  --max_material_object_id 2


# Add the trained Object0 PB to every timestep of the known-to-operator A dataset
python robo_manip_baselines/policy/wrench_predictor4_online/AddConstantPbToDataset.py \
  robo_manip_baselines/dataset/LiftingAB_B_only_ConstantPb \
  --checkpoint robo_manip_baselines/checkpoint/WrenchPredictor4/LiftingAB_B_only/policy_last.ckpt

# Estimate PB online from each unknown-to-operator B episode, starting from Object0 PB
uv run python robo_manip_baselines/policy/wrench_predictor4_online/AddOnlinePbToDataset.py \
  robo_manip_baselines/dataset/LiftingAB_B_only/ 0 \
  --checkpoint robo_manip_baselines/checkpoint/WrenchPredictor4/LiftingAB_B_only/policy_last.ckpt \
  --lr 6e-3

# Plot the online PB trajectories stored in the Object1 B episodes
uv run python robo_manip_baselines/misc/futureimagination/PlotOnlinePbDataset.py \
  robo_manip_baselines/dataset/LiftingAB_B_only/WrenchPredObject1

# Add deterministic SD3 VAE features from every left-camera frame
python robo_manip_baselines/misc/futureimagination/AddImageFeature.py \
  robo_manip_baselines/dataset/LiftingAB_B_only \
  --overwrite

# Train state-based Diffusion Policy on all 150 LiftingAB A/B episodes with PB
uv run python robo_manip_baselines/bin/Train.py DiffusionPolicy \
  --dataset_dir robo_manip_baselines/dataset/LiftingAB_B_only \
  --camera_names \
  --state_keys measured_eef_pose measured_gripper_joint_pos measured_tblock_pose material_property \
  --action_keys command_eef_pose command_gripper_joint_pos \
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

# Train MLP-only WrenchPredictor4 on the 75 disjoint unknown-to-operator B episodes
uv run python robo_manip_baselines/bin/Train.py WrenchPredictor4 \
  --dataset_dir robo_manip_baselines/dataset/LiftingAB_B_only \
  --checkpoint_dir robo_manip_baselines/checkpoint/WrenchPredictor4/LiftingAB_B_only \
  --camera_names \
  --state_keys measured_eef_pose measured_gripper_joint_pos \
  --action_keys command_eef_pose command_gripper_joint_pos \
  --image_feature_key measured_tblock_pose \
  --wrench_source_key measured_eef_wrench \
  --pb_dim 1 \
  --output_head mlp_only \
  --wrench_loss_weight 0.1 \
  --scheduler ddpm \
  --horizon 16 \
  --n_obs_steps 2 \
  --skip 3 \
  --batch_size 64 \
  --num_epochs 500 \
  --lr 1e-4 \
  --train_ratio 1.0 \
  --val_ratio 0.01

# Roll out Diffusion Policy while adapting PB online with WrenchPredictor4
uv run python robo_manip_baselines/bin/Rollout.py \
  DiffusionPolicyOnlinePb MujocoUR5eLiftingi_I1 \
  --checkpoint path/to/diffusion_policy.ckpt \
  --wp4_checkpoint path/to/wrench_predictor4.ckpt \
  --initial_object_id 0 \
  --online_pb_lr 6e-3


#Rollout for evaluation of Diffusion Policy with online PB adaptation

python robo_manip_baselines/bin/Rollout.py \
  DiffusionPolicyOnlinePb MujocoUR5eLiftingi_I0 \
  --checkpoint path/to/diffusion_policy.ckpt \
  --wp4_checkpoint robo_manip_baselines/checkpoint/WrenchPredictor4/LiftingAB_B_only/policy_last.ckpt \
  --initial_object_id 0 \
  --online_pb_lr 6e-3 \
  --world_idx_list {70..79} --auto_exit --max_duration 10 --save_rollout --no_plot ;

python robo_manip_baselines/bin/Rollout.py \
  DiffusionPolicyOnlinePb MujocoUR5eLiftingi_I0 \
  --checkpoint path/to/diffusion_policy.ckpt \
  --wp4_checkpoint robo_manip_baselines/checkpoint/WrenchPredictor4/LiftingAB_B_only/policy_last.ckpt \
  --initial_object_id 0 \
  --online_pb_lr 6e-3 \
  --world_idx_list {70..79} --auto_exit --max_duration 10 --save_rollout --no_plot ;

python robo_manip_baselines/bin/Rollout.py \
  DiffusionPolicyOnlinePb MujocoUR5eLiftingi_I1 \
  --checkpoint path/to/diffusion_policy.ckpt \
  --wp4_checkpoint robo_manip_baselines/checkpoint/WrenchPredictor4/LiftingAB_B_only/policy_last.ckpt \
  --initial_object_id 0 \
  --online_pb_lr 6e-3 \
  --world_idx_list {170..179} --auto_exit --max_duration 10 --save_rollout --no_plot ;

python robo_manip_baselines/bin/Rollout.py \
  DiffusionPolicyOnlinePb MujocoUR5eLiftingi_I2 \
  --checkpoint path/to/diffusion_policy.ckpt \
  --wp4_checkpoint robo_manip_baselines/checkpoint/WrenchPredictor4/LiftingAB_B_only/policy_last.ckpt \
  --initial_object_id 0 \
  --online_pb_lr 6e-3 \
  --world_idx_list {270..279} --auto_exit --max_duration 10 --save_rollout --no_plot ;


# Judge all 60 Lifting test episodes: final lift >= 10 cm and tilt < 7.5 deg
uv run python robo_manip_baselines/misc/futureimagination/AnalyzeLiftingSuccess.py \
  robo_manip_baselines/dataset/tests/FutureImagination/LiftingAB_DP \
  --output_prefix validation_success \
  --lift_threshold_m 0.10 \
  --tilt_threshold_deg 7.5 \
  --expected_episode_count 60


#####################################################################################








# Baseline: ######################################################################################
# train state-based Diffusion Policy on the 75 unknown-to-operator B episodes without PB
uv run python robo_manip_baselines/bin/Train.py DiffusionPolicy \
  --dataset_dir robo_manip_baselines/dataset/LiftingAB_B_only \
  --checkpoint_dir robo_manip_baselines/checkpoint/DiffusionPolicy/LiftingAB_B_75_state_tblock_no_pb_cnn_ddim_h16_e500_20260803 \
  --camera_names \
  --state_keys measured_eef_pose measured_gripper_joint_pos measured_tblock_pose \
  --action_keys command_eef_pose command_gripper_joint_pos \
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


# Rollout for evaluation #

uv run python robo_manip_baselines/bin/Rollout.py DiffusionPolicy MujocoUR5eLiftingi_I0 \
 --checkpoint robo_manip_baselines/checkpoint/DiffusionPolicy/LiftingAB_B75_policy_last_seed42_52_62/baselines/seed42/policy_last.ckpt \
 --world_idx_list {70..79} --auto_exit --max_duration 10 --save_rollout --no_plot ;
 uv run python robo_manip_baselines/bin/Rollout.py DiffusionPolicy MujocoUR5eLiftingi_I1 \
  --checkpoint robo_manip_baselines/checkpoint/DiffusionPolicy/LiftingAB_B75_policy_last_seed42_52_62/baselines/seed42/policy_last.ckpt \
  --world_idx_list {170..179} --auto_exit --max_duration 10 --save_rollout --no_plot ;
  uv run python robo_manip_baselines/bin/Rollout.py DiffusionPolicy MujocoUR5eLiftingi_I2\
 --checkpoint robo_manip_baselines/checkpoint/DiffusionPolicy/LiftingAB_B75_policy_last_seed42_52_62/baselines/seed42/policy_last.ckpt \
 --world_idx_list {270..279} --auto_exit --max_duration 10 --save_rollout --no_plot

##############################################################################################################

# Blind teleoperation collection (trial counts are configured in the script)
uv run python unstructured/collect_blind_teleop.py
