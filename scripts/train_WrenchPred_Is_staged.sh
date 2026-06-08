#!/bin/bash
#PBS -q rt_HG
#PBS -l select=1
#PBS -l walltime=60:00:00
#PBS -P gag51454

set -eu

source ~/miniconda3/bin/activate
conda activate rmb

source ~/projects/ForceImagination/RoboManipBaselines/.venv/bin/activate

cd ~/projects/ForceImagination/RoboManipBaselines/

STAGE1_DATASET_DIR=dataset/DatasetIs/traing_stage1
STAGE2_DATASET_DIR=dataset/DatasetIs/traning_stage2

STAGE1_CHECKPOINT_DIR=robo_manip_baselines/checkpoint/WrenchPredictor3/DatasetIs_stage1
STAGE2_CHECKPOINT_DIR=robo_manip_baselines/checkpoint/WrenchPredictor3/DatasetIs_stage2_material_only

python robo_manip_baselines/bin/Train.py WrenchPredictor3 \
    --dataset_dir ${STAGE1_DATASET_DIR} \
    --checkpoint_dir ${STAGE1_CHECKPOINT_DIR} \
    --state_keys measured_eef_pose measured_eef_pose_rel \
    --action_keys command_eef_pose \
    --lr 3e-5 \
    --lr_material_property 1e-3 \
    --lr_backbone 1e-5 \
    --num_epochs 100 \
    --batch_size 64 \
    --camera_names front hand hand_2

python robo_manip_baselines/bin/Train.py WrenchPredictor3 \
    --dataset_dir ${STAGE2_DATASET_DIR} \
    --checkpoint_dir ${STAGE2_CHECKPOINT_DIR} \
    --pretrain_checkpoint ${STAGE1_CHECKPOINT_DIR}/policy_last.ckpt \
    --state_keys measured_eef_pose measured_eef_pose_rel \
    --action_keys command_eef_pose \
    --lr 0 \
    --lr_material_property 1e-3 \
    --lr_backbone 0 \
    --num_epochs 100 \
    --batch_size 64 \
    --camera_names front hand hand_2
