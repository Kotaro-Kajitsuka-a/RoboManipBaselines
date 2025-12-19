

================================================================================
# Go to the top directory of this repository
cd robo_manip_baselines
# Connect a SpaceMouse to your PC
python ./bin/Teleop.py MujocoXarm7Ring --world_idx_list 0 5 --input_device keyboard
================================================================================




================================================================================
# Go to the top directory of this repository
cd robo_manip_baselines
python ./bin/Rollout.py Act MujocoUR5eCable \
--checkpoint robo_manip_baselines/checkpoint/Act/training_data_Act_20250908_154926/policy_last.ckpt \
--world_idx 0
================================================================================



Act
==========================================================================================================
python ./bin/Teleop.py RealXarm7Demo --config ./envs/configs/RealXarm7DemoEnv.yaml --input_device spacemouse
python ./bin/Train.py Act --dataset_dir ./dataset/RealXarm7Demo_20 --num_epochs 250
python ./bin/Rollout.py Act RealXarm7Demo --config ./envs/configs/RealXarm7DemoEnv.yaml --checkpoint ./checkpoint/Act/RealXarm7Demo_20250911_162223_Act_20250911_163110/policy_best.ckpt --wait_before_start
====================================================================================================

Mlp
==========================================================================================================
python ./robo_manip_baselines/bin/Teleop.py RealXarm7Demo --config ./robo_manip_baselines/envs/configs/RealXarm7DemoEnv.yaml --input_device spacemouse
python ./bin/Train.py Mlp --dataset_dir ./dataset/RealXarm7Demo_20250911_162223 --num_epochs 250
python ./bin/Rollout.py Mlp RealXarm7Demo --config ./envs/configs/RealXarm7DemoEnv.yaml --checkpoint ./checkpoint/Mlp/RealXarm7Demo_20250911_162223_Mlp_20250922_162104/policy_epoch175.ckpt --wait_before_start
no plotでも以外と重い。↓
python ./bin/Rollout.py Mlp RealXarm7Demo --config ./envs/configs/RealXarm7DemoEnv.yaml --checkpoint ./checkpoint/Mlp/RealXarm7Demo_20250911_162223_Mlp_20250922_162104/policy_epoch175.ckpt  --no_plot
====================================================================================================

Ppo
python ./bin/Rollout.py Ppo RealXarm7Demo --wait_before_start --skip_draw 50 --save_rollout --config ./envs/configs/RealXarm7DemoEnv.yaml --checkpoint final_ckpt.pt 

Ppo_cus
python ./bin/Rollout.py PpoCus RealXarm7Demo --wait_before_start --skip_draw 50 --ppo-marker-enable --ppo-enable-vision --save_rollout --config ./envs/configs/RealXarm7DemoEnv.yaml --checkpoint final_ckpt.pt


visualize
python ./bin/ViewModelMetaInfo.py ./checkpoint/Mlp/RealXarm7Demo_20250911_162223_Mlp_20250922_162104/model_meta_info.pkl --save-json meta_info_info.json


make pickle file of meta_info
python ./bin/CreatePpoCusMetaInfo.py --config ./ppo_tasks/jointhold_marker_check.json --output ./checkpoint/PpoCus/Align/model_meta_info.pkl /
python ./bin/CreatePpoCusMetaInfo.py --config ./ppo_tasks/align.json --output ./checkpoint/PpoCus/Align/model_meta_info_align.pkl --force

extrinsic_calibration
python robo_manip_baselines/bin/extrinsic_calibraition.py

make pickle and run
##########################################################################
python ./bin/CreatePpoCusMetaInfo.py \
  --config ./ppo_tasks/jointhold_marker_check.json \
  --output ./checkpoint/PpoCus/Align/model_meta_info.pkl \
  --force

python ./robo_manip_baselines/bin/Rollout.py PpoCus RealXarm7Demo \
  --wait_before_start \
  --skip_draw 500000 \
  --save_rollout \
  --config ./robo_manip_baselines/envs/configs/RealXarm7DemoEnv.yaml \
  --checkpoint final_ckpt.pt
############################################################################


make pickle and run(align)
##########################################################################
python ./bin/CreatePpoCusMetaInfo.py \
  --config ./ppo_tasks/align.json \
  --output ./checkpoint/PpoCus/Align/model_meta_info.pkl \
  --force

python ./robo_manip_baselines/bin/Rollout.py PpoCus RealXarm7Demo \
  --wait_before_start \
  --skip_draw 50000 \
  --save_rollout \
  --config ./robo_manip_baselines/envs/configs/RealXarm7DemoEnv.yaml \
  --checkpoint ./robo_manip_baselines/checkpoint/PpoCus/Align/002_351.pt
############################################################################


make pickle and run(DualBoxRotationAblated)
##########################################################################
python ./robo_manip_baselines/bin/CreatePpoCusMetaInfo.py \
  --config ./robo_manip_baselines/ppo_tasks/dual_box_rotation_ablated.json \
  --output ./robo_manip_baselines/checkpoint/PpoCus/DualBoxRotationAblated/model_meta_info.pkl \
  --force

python ./robo_manip_baselines/bin/Rollout.py PpoCus RealXarm7DualDemo \
  --wait_before_start \
  --skip_draw 50000 \
  --save_rollout \
  --config ./robo_manip_baselines/envs/configs/RealXarm7DualDemoEnv.yaml \
  --checkpoint ./robo_manip_baselines/checkpoint/PpoCus/DualBoxRotationAblated/ckpt_126.pt
############################################################################

make pickle and run(DualSimple)
##########################################################################
python .//bin/CreatePpoCusMetaInfo.py \
  --config ./ppo_tasks/dual_simple.json \
  --output ./checkpoint/PpoCus/DualSimple/model_meta_info.pkl \
  --force

python ./robo_manip_baselines/bin/Rollout.py PpoCus RealXarm7DualDemo \
  --wait_before_start \
  --skip_draw 50000 \
  --save_rollout \
  --config ./robo_manip_baselines/envs/configs/RealXarm7DualDemoEnv.yaml \
  --checkpoint ./robo_manip_baselines/checkpoint/PpoCus/DualSimple/ckpt_26.pt
############################################################################

python ./bin/CollectXarm7Dynamics.py --rojljbot-ip 192.168.1.244 --duration 120 --sample-rate 200 --output-dir ./measurements
l

#双腕teleop
python ./robo_manip_baselines/bin/Teleop.py RealXarm7DualDemo --config ./robo_manip_baselines/envs/configs/RealXarm7DualDemoEnv.yaml --input_device keyboard


#box 検出単体
python ./robo_manip_baselines/policy/ppo_cus/ppo_tasks/dual_box_rotation_ablated.py 
python ./robo_manip_baselines/policy/ppo_cus/ppo_tasks/box_pose_viewer.py

#box 検出単体(SAC)
python ./robo_manip_baselines/policy/sac/sac_tasks/dual_box_rotation.py 
python ./robo_manip_baselines/policy/sac/sac_tasks/box_pose_viewer.py

#error code確認
python robo_manip_baselines/bin/check_xarm_err.py --ip-left 192.168.1.244 --ip-right 192.168.1.211

#set
python robo_manip_baselines/bin/xarm_dual_gravity_comp.py \
  --ip-left 192.168.1.244 --ip-right 192.168.1.211



make pickle and run(DualBoxRotation)
##########################################################################
python ./robo_manip_baselines/bin/CreatePpoCusMetaInfo.py \
  --config ./robo_manip_baselines/ppo_tasks/dual_box_rotation.json \
  --output ./robo_manip_baselines/checkpoint/Sac/DualBoxRotation/model_meta_info.pkl \
  --force

python ./robo_manip_baselines/bin/Rollout.py Sac RealXarm7DualDemo \
  --wait_before_start \
  --skip_draw 50000 \
  --save_rollout \
  --config ./robo_manip_baselines/envs/configs/RealXarm7DualDemoEnv.yaml \
  --checkpoint robo_manip_baselines/checkpoint/Sac/DualBoxRotation/ckpt_2750016.pt \
  --world_idx_repeat_count 10 

python ./robo_manip_baselines/bin/Train.py Act \
  --dataset_dir robo_manip_baselines/dataset/26epi_RolloutSac_RealXarm7DualDemo_20251219_145609 \
  --num_epochs 250 --camera_names front left_hand right_hand

############################################################################




Act
==========================================================================================================
python ./robo_manip_baselines/bin/Teleop.py RealXarm7DualDemo --config ./robo_manip_baselines/envs/configs/RealXarm7DualDemoEnv.yaml --input_device keyboard
python ./robo_manip_baselines/bin/Train.py Act --dataset_dir robo_manip_baselines/dataset/RealXarm7DualDemo_20251218_154553 --num_epochs 250 --camera_names left_hand right_hand
python ./robo_manip_baselines/bin/Rollout.py Act RealXarm7DualDemo --config ./robo_manip_baselines/envs/configs/RealXarm7DualDemoEnv.yaml --checkpoint robo_manip_baselines/checkpoint/Act/RealXarm7DualDemo_20251218_154553_Act_20251218_154950/policy_epoch125.ckpt --wait_before_start --skip_draw 50000 --save_rollout --world_idx_repeat_count 10
====================================================================================================

Train from Act Rollout and rollout imitation policy of imitation policy
==============================================================================================================
python ./robo_manip_baselines/bin/Train.py Act --dataset_dir robo_manip_baselines/dataset/RolloutAct_RealXarm7DualDemo_20251219_142326 --num_epochs 250 --camera_names left_hand right_hand
python ./robo_manip_baselines/bin/Rollout.py Act RealXarm7DualDemo --config ./robo_manip_baselines/envs/configs/RealXarm7DualDemoEnv.yaml --checkpoint robo_manip_baselines/checkpoint/Act/RolloutAct_RealXarm7DualDemo_20251219_142326_Act_20251219_143133/policy_best.ckpt --wait_before_start --skip_draw 50000 --save_rollout --world_idx_repeat_count 10
==============================================================================================================