python robo_manip_baselines/policy/rl_policy_dual/rl_tasks/view_single_aruco_marker.py





#########################################################################

python ./robo_manip_baselines/bin/CreatePpoCusMetaInfo.py \
  --config ./robo_manip_baselines/rl_tasks/trash_bin_rolling.json \
  --output "robo_manip_baselines/checkpoint/PpoCus/TrashBinRolling/model_meta_info.pkl" \
  --force

python ./robo_manip_baselines/bin/Rollout.py Sac RealXarm7DualFixedGripperDemo \
  --checkpoint robo_manip_baselines/checkpoint/PpoCus/TrashBinRolling/0714_ckpt_471.pt \
  --wait_before_start \
  --skip_draw 50000 \
  --save_rollout \
  --config ./robo_manip_baselines/envs/configs/RealXarm7DualDemoEnv.yaml \
  --world_idx_repeat_count 45

###########################################################################
