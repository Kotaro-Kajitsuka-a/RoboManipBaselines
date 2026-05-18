#!/bin/bash
#PBS -q rt_HG                                                                                                                                
#PBS -l select=1                                                                                                                             
#PBS -l walltime=30:00:00                                                                                                                    
#PBS -P gag51454  

set -eu

source ~/miniconda3/bin/activate
conda activate rmb

source ~/projects/ForceImagination/RoboManipBaselines/.venv/bin/activate

cd ~/projects/ForceImagination/RoboManipBaselines/

CHECKPOINT_DIR=robo_manip_baselines/checkpoint/WrenchPredictor3/DatasetColl_320x240

python robo_manip_baselines/bin/Train.py WrenchPredictor3 \
    --dataset_dir robo_manip_baselines/dataset/DatasetColl/training \
    --checkpoint_dir ${CHECKPOINT_DIR} \
    --state_keys measured_eef_pose measured_eef_pose_rel \
    --action_keys command_eef_pose \
    --chunk_size 1 \
    --lr 3e-5 \
    --num_epochs 100 \
    --batch_size 64 \
    --camera_names front hand hand_2

python robo_manip_baselines/policy/wrench_predictor3/EvalWrenchPredictorDir.py \
    ${CHECKPOINT_DIR}/policy_last.ckpt \
    robo_manip_baselines/dataset/DatasetColl/validation/lying_id

python robo_manip_baselines/policy/wrench_predictor3/EvalWrenchPredictorDir.py \
    ${CHECKPOINT_DIR}/policy_last.ckpt \
    robo_manip_baselines/dataset/DatasetColl/validation/normal

python robo_manip_baselines/policy/wrench_predictor3/EvalWrenchPredictorDir.py \
    ${CHECKPOINT_DIR}/policy_last.ckpt \
    robo_manip_baselines/dataset/DatasetColl/validation/lying_id_0123

python robo_manip_baselines/policy/wrench_predictor3/EvalWrenchPredictorDir.py \
    ${CHECKPOINT_DIR}/policy_last.ckpt \
    robo_manip_baselines/dataset/DatasetColl/validation/normal_0123
