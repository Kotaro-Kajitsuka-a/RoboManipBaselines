python ./robo_manip_baselines/policy/ppo_cus/ppo_tasks/dual_box_rotation_ablated.py
python ./robo_manip_baselines/policy/ppo_cus/ppo_tasks/box_pose_viewer.py


#Maniskillで学習したポリシーのロールアウト
#########################################################################

python ./robo_manip_baselines/bin/CreatePpoCusMetaInfo.py \
  --config ./robo_manip_baselines/rl_tasks/single_cardboard_cabinet.json \
  --output "robo_manip_baselines/checkpoint/PpoCus/OpenDrawer/model_meta_info.pkl" \
  --force

python ./robo_manip_baselines/bin/Rollout.py RLPolicy RealXarm7Demo \
  --checkpoint "robo_manip_baselines/checkpoint/PpoCus/OpenDrawer/ckpt_41.pt" \
  --save_rollout \
  --wait_before_start \
  --skip_draw 50000 \
  --config ./robo_manip_baselines/envs/configs/RealXarm7DemoEnv.yaml \
  --world_idx_repeat_count 45

###########################################################################
