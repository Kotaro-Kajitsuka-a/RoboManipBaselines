python robo_manip_baselines/policy/rl_policy_dual/rl_tasks/view_single_aruco_marker.py





#########################################################################

python ./robo_manip_baselines/bin/CreatePpoCusMetaInfo.py \
  --config ./robo_manip_baselines/rl_tasks/dual_joint_pos_vel_marker_fixed_gripper.json \
  --output "robo_manip_baselines/checkpoint/PpoCus/TrashBinRolling/model_meta_info.pkl" \
  --force

python ./robo_manip_baselines/bin/Rollout.py RLPolicyDual RealXarm7DualDemo \
  --checkpoint robo_manip_baselines/checkpoint/PpoCus/TrashBinRolling/ckpt_41.pt \
  --wait_before_start \
  --skip_draw 50000 \
  --config ./robo_manip_baselines/envs/configs/RealXarm7DualDemoEnvTrashBinRolling.yaml \
  --world_idx_repeat_count 45

###########################################################################
