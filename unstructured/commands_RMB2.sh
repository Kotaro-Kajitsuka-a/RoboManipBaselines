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
  --action_keys command_eef_pose command_gripper_joint_pos \
  --dataset_dir ./dataset/MujocoPusht_012 \
  --scheduler ddim --num_epochs 1000 \
  --train_ratio 1.0 --val_ratio 0.01

python ./robo_manip_baselines/bin/Rollout.py DiffusionPolicy MujocoXarm7AdmittancePusht \
  --checkpoint robo_manip_baselines/checkpoint/DiffusionPolicy/ \
  --wait_before_start --skip_draw 50000 --save_rollout --world_idx_repeat_count 10
