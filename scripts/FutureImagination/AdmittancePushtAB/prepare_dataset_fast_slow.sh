#!/usr/bin/env bash
set -e

# Run from the repository root after train_wp4.sh. Keep the source data
# untouched. Fast training episodes receive learned constant PBs, while slow
# training and all validation episodes receive causal online-PB trajectories.
fast_source=robo_manip_baselines/dataset/DatasetAdmittancePushtAB/training/fast
slow_source=robo_manip_baselines/dataset/DatasetAdmittancePushtAB/training/slow
validation_source=robo_manip_baselines/dataset/DatasetAdmittancePushtAB/validation
fast_slow_dataset=robo_manip_baselines/dataset/DatasetAdmittancePushtAB_fast_slow
validation_online_dataset=robo_manip_baselines/dataset/DatasetAdmittancePushtAB_validation_online_pb
wp4_checkpoint=robo_manip_baselines/checkpoint/WrenchPredictor4/AdmittancePushtAB/policy_best.ckpt

mkdir -p \
  "$fast_slow_dataset/fast" \
  "$fast_slow_dataset/slow" \
  "$validation_online_dataset"

cp -aL "$fast_source/." "$fast_slow_dataset/fast/"
cp -aL "$slow_source/." "$fast_slow_dataset/slow/"
cp -aL "$validation_source/." "$validation_online_dataset/"

python robo_manip_baselines/policy/wrench_predictor4_online/AddConstantPbToDataset.py \
  "$fast_slow_dataset/fast" \
  --checkpoint "$wp4_checkpoint" \
  --overwrite

python robo_manip_baselines/policy/wrench_predictor4_online/AddOnlinePbToDataset.py \
  "$fast_slow_dataset/slow" \
  0 \
  --checkpoint "$wp4_checkpoint" \
  --lr 2.5e-2 \
  --wrench_loss_weight 0.01 \
  --overwrite

python robo_manip_baselines/policy/wrench_predictor4_online/AddOnlinePbToDataset.py \
  "$validation_online_dataset" \
  0 \
  --checkpoint "$wp4_checkpoint" \
  --lr 2.5e-2 \
  --wrench_loss_weight 0.01 \
  --overwrite

python robo_manip_baselines/misc/futureimagination/PlotOnlinePbDataset2d.py \
  "$fast_slow_dataset/slow/WrenchPredObject0" \
  --material_object_ids 0 1 2 3

python robo_manip_baselines/misc/futureimagination/PlotOnlinePbDataset2d.py \
  "$fast_slow_dataset/slow/WrenchPredObject1" \
  --material_object_ids 0 1 2 3

python robo_manip_baselines/misc/futureimagination/PlotOnlinePbDataset2d.py \
  "$fast_slow_dataset/slow/WrenchPredObject2" \
  --material_object_ids 0 1 2 3

python robo_manip_baselines/misc/futureimagination/PlotOnlinePbDataset2d.py \
  "$fast_slow_dataset/slow/WrenchPredObject3" \
  --material_object_ids 0 1 2 3

python robo_manip_baselines/misc/futureimagination/PlotOnlinePbDataset2d.py \
  "$validation_online_dataset/WrenchPredObject0" \
  --material_object_ids 0 1 2 3

python robo_manip_baselines/misc/futureimagination/PlotOnlinePbDataset2d.py \
  "$validation_online_dataset/WrenchPredObject1" \
  --material_object_ids 0 1 2 3

python robo_manip_baselines/misc/futureimagination/PlotOnlinePbDataset2d.py \
  "$validation_online_dataset/WrenchPredObject2" \
  --material_object_ids 0 1 2 3

python robo_manip_baselines/misc/futureimagination/PlotOnlinePbDataset2d.py \
  "$validation_online_dataset/WrenchPredObject3" \
  --material_object_ids 0 1 2 3
