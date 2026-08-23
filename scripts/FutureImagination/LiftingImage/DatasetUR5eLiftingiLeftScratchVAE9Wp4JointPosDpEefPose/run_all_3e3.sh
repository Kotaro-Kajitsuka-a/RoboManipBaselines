#!/usr/bin/env bash
set -e

# Adam-only online PB experiment with learning rate 3e-3. The suffix keeps its
# datasets, DP checkpoints, and rollout results separate from the 6e-3 run.
export ONLINE_PB_LR=3e-3
export EXPERIMENT_SUFFIX=_lr3e3

bash scripts/FutureImagination/LiftingImage/DatasetUR5eLiftingiLeftScratchVAE9Wp4JointPosDpEefPose/run_all.sh
