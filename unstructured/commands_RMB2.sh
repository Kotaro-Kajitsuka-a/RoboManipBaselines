#Teleopration with vive
python ./robo_manip_baselines/bin/Teleop.py MujocoXarm7DualAdmittanceBox --input_device_config --input_device vive ./robo_manip_baselines/teleop/configs/ViveDual.yaml --world_idx_list 0 1


#ただのteleopration
python robo_manip_baselines/bin/Teleop.py MujocoXarm7AdmittancePusht


#replay_log
python robo_manip_baselines/bin/Teleop.py MujocoXarm7AdmittancePusht \
  --ignore_replay_world_idx --replay_log \
  robo_manip_baselines/dataset/MujocoXarm7Pusht_20260420_134319/MujocoXarm7Pusht_world20_002.rmb/ \
  --world_idx_list 10
