python ./robo_manip_baselines/policy/ppo_cus/ppo_tasks/dual_box_rotation_ablated.py
python ./robo_manip_baselines/policy/ppo_cus/ppo_tasks/box_pose_viewer.py


#Maniskillで学習したポリシーのロールアウト
#########################################################################
CHECKPOINT=/path/to/cabinet_ppo_checkpoint.pt
CHECKPOINT_DIR=$(dirname "${CHECKPOINT}")

uv run python ./robo_manip_baselines/bin/CreatePpoCusMetaInfo.py \
  --config ./robo_manip_baselines/rl_tasks/single_cardboard_cabinet.json \
  --output "${CHECKPOINT_DIR}/model_meta_info.pkl" \
  --force

uv run python ./robo_manip_baselines/bin/Rollout.py RLPolicy RealXarm7Demo \
  --checkpoint "${CHECKPOINT}" \
  --save_rollout \
  --wait_before_start \
  --skip_draw 50000 \
  --config ./robo_manip_baselines/envs/configs/RealXarm7DemoEnv.yaml \
  --world_idx_repeat_count 45

###########################################################################
