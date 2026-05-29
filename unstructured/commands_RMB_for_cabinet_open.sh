python ./robo_manip_baselines/policy/rl_policy/rl_tasks/single_cardboard_cabinet.py


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
  --config ./robo_manip_baselines/envs/configs/RealXarm7DemoEnvCabinetOpen.yaml \
  --world_idx_repeat_count 45

###########################################################################
