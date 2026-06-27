#Teleopration with vive
python ./robo_manip_baselines/bin/Teleop.py MujocoXarm7DualAdmittanceBox --input_device vive --input_device_config  ./robo_manip_baselines/teleop/configs/ViveDual.yaml --world_idx_list 0 1


#ただのteleopration
python robo_manip_baselines/bin/Teleop.py MujocoXarm7AdmittancePusht

python robo_manip_baselines/bin/Teleop.py MujocoXarm7AdmittancePushi_I0 --input_device keyboard --world_idx_list 0 1 2 3

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

# Add frozen Diffusion Policy visual features to all RMB HDF5 files
uv run python robo_manip_baselines/policy/diffusion_world_model/SaveObsFeatures.py \
  robo_manip_baselines/checkpoint/DiffusionPolicy/Stage0_DatasetAdmittancePushtRandom_cnn_ddim_seed62_eef_pose/policy_epoch0000.ckpt \
  robo_manip_baselines/dataset/DatasetAdmittancePushtRandom \
  --overwrite

# Train DiffusionWorldModel with frozen Diffusion Policy visual features
uv run python robo_manip_baselines/bin/Train.py DiffusionWorldModel \
  --dataset_dir robo_manip_baselines/dataset/DatasetColl \
  --state_keys measured_eef_pose \
  --action_keys command_eef_pose \
  --scheduler ddim \
  --horizon 16 \
  --n_obs_steps 2 \
  --wrench_source_key measured_eef_wrench_moving_average \
  --pb_dim 9 \
  --num_epochs 1000 \
  --batch_size 64 \
  --train_ratio 0.99 \
  --val_ratio 0.01 

# Train DiffusionWorldModel with AprilTag pose instead of frozen visual features
uv run python robo_manip_baselines/bin/Train.py DiffusionWorldModel \
  --dataset_dir robo_manip_baselines/dataset/ピンチテスト_marker/training \
  --state_keys measured_eef_pose \
  --action_keys command_eef_pose \
  --image_feature_key front_apriltag_pose_xy_axis \
  --wrench_source_key measured_eef_wrench \
  --scheduler ddim \
  --horizon 16 \
  --n_obs_steps 2 \
  --pb_dim 9 \
  --num_epochs 1000 \
  --batch_size 64 \
  --train_ratio 0.99 \
  --val_ratio 0.01

# Evaluate all DiffusionWorldModel checkpoints with material-property sweep
uv run python robo_manip_baselines/policy/diffusion_world_model/EvalDiffusionWorldModelMaterialSweepDir.py \
  robo_manip_baselines/checkpoint/DiffusionWorldModel/hogehoge \
  robo_manip_baselines/dataset/DatasetColl/validation

# Add AprilTag tag25h9 pose to all RMB HDF5 files
uv run python robo_manip_baselines/misc/AddAprilTagPoseToRmbData.py \
  robo_manip_baselines/dataset/ピンチテスト_marker/training \
  --camera_name front \
  --tag_size 0.03385 \
  --tag_id 0 \
  --save_video \
  --overwrite
