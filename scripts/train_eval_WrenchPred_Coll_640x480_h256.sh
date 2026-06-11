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

TRAIN_DATASET_DIR=robo_manip_baselines/dataset/DatasetColl/training
EVAL_DIR=robo_manip_baselines/dataset/DatasetColl/validation/normal_0123

CHECKPOINT_ROOT=robo_manip_baselines/checkpoint/WrenchPredictor3
CHECKPOINT_DIR=${CHECKPOINT_ROOT}/DatasetColl_640x480_h256

NUM_EPOCHS=${NUM_EPOCHS:-100}
BATCH_SIZE=${BATCH_SIZE:-32}

COMMON_ARGS=(
    --state_keys measured_eef_pose measured_eef_pose_rel
    --action_keys command_eef_pose
    --image_width 640
    --image_height 480
    --hidden_dim 256
    --dim_feedforward 1024
    --num_epochs "${NUM_EPOCHS}"
    --batch_size "${BATCH_SIZE}"
    --camera_names front hand hand_2
)

python robo_manip_baselines/bin/Train.py WrenchPredictor3 \
    --dataset_dir ${TRAIN_DATASET_DIR} \
    --checkpoint_dir ${CHECKPOINT_DIR} \
    --lr 3e-5 \
    --lr_material_property 1e-3 \
    --lr_backbone 1e-5 \
    "${COMMON_ARGS[@]}"

python robo_manip_baselines/policy/wrench_predictor3/EvalWrenchPredictorMaterialSweepDir.py \
    ${CHECKPOINT_DIR}/policy_last.ckpt \
    ${EVAL_DIR}
