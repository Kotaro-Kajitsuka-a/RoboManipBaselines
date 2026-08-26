  #!/usr/bin/env bash
  set -e

  # Add PB labels to copies of the new dataset's training split.
  source_dataset=robo_manip_baselines/dataset/0826_DatasetSorting/training/
  online_dataset=robo_manip_baselines/dataset/0826_DatasetSorting_FrontScratchVAE9_OnlinePB/training/
  constant_dataset=robo_manip_baselines/dataset/0826_DatasetSorting_FrontScratchVAE9_ConstantPB/training/
  wp4_checkpoint=robo_manip_baselines/checkpoint/WrenchPredictor4/0826_DatasetSorting_front_image_vae_9_joint_pos/policy_best.ckpt

  mkdir -p "$online_dataset" "$constant_dataset"
  cp -aL "$source_dataset/." "$online_dataset/"
  cp -aL "$source_dataset/." "$constant_dataset/"

  python robo_manip_baselines/policy/wrench_predictor4_online/AddOnlinePbToDataset.py \
    "$online_dataset" \
    1 \
    --checkpoint "$wp4_checkpoint" \
    --lr 8e-3 \
    --wrench_loss_weight 0.0 \
    --overwrite

  python robo_manip_baselines/policy/wrench_predictor4_online/AddConstantPbToDataset.py \
    "$constant_dataset" \
    --checkpoint "$wp4_checkpoint" \
    --overwrite



  python robo_manip_baselines/misc/futureimagination/PlotOnlinePbDataset.py \
    "$online_dataset/WrenchPredObject0"

  python robo_manip_baselines/misc/futureimagination/PlotOnlinePbDataset.py \
    "$online_dataset/WrenchPredObject1"

  python robo_manip_baselines/misc/futureimagination/PlotOnlinePbDataset.py \
    "$online_dataset/WrenchPredObject2"


  # The following is labeling for the validation data.

  source_validation_dataset=robo_manip_baselines/dataset/0826_DatasetSorting/validation/
  online_validation_dataset=robo_manip_baselines/dataset/0826_DatasetSorting_FrontScratchVAE9_OnlinePB/validation/

  mkdir -p "$online_validation_dataset"
  cp -aL "$source_validation_dataset/." "$online_validation_dataset/"

  python robo_manip_baselines/policy/wrench_predictor4_online/AddOnlinePbToDataset.py \
    "$online_validation_dataset" \
    1 \
    --checkpoint "$wp4_checkpoint" \
    --lr 8e-3 \
    --wrench_loss_weight 0.0 \
    --overwrite


  python robo_manip_baselines/misc/futureimagination/PlotOnlinePbDataset.py \
    "$online_validation_dataset/WrenchPredObject0"

  python robo_manip_baselines/misc/futureimagination/PlotOnlinePbDataset.py \
    "$online_validation_dataset/WrenchPredObject1"

  python robo_manip_baselines/misc/futureimagination/PlotOnlinePbDataset.py \
    "$online_validation_dataset/WrenchPredObject2"
