#!/bin/bash
#PBS -q rt_HG
#PBS -l select=1
#PBS -l walltime=10:00:00
#PBS -P gag51454

set -eu

source ~/miniconda3/bin/activate
conda activate rmb

source ~/projects/ForceImagination/RoboManipBaselines/.venv/bin/activate

cd ~/projects/ForceImagination/RoboManipBaselines/

CHECKPOINT=${CHECKPOINT:-robo_manip_baselines/checkpoint/WrenchPredictor3/DatasetColl_640x480_h256/policy_last.ckpt}
EVAL_DIR=${EVAL_DIR:-robo_manip_baselines/dataset/DatasetColl/validation/normal_0123}

python robo_manip_baselines/policy/wrench_predictor3/EvalWrenchPredictorMaterialSweepDir.py \
    ${CHECKPOINT} \
    ${EVAL_DIR}
