#!/usr/bin/env bash
# Run all six conditions from the repository root, continuing after failures.
bash scripts/PressureSensor/knife/train_baseline_right.sh
bash scripts/PressureSensor/knife/train_baseline_right_pressure.sh
bash scripts/PressureSensor/knife/train_baseline_right_hand.sh
bash scripts/PressureSensor/knife/train_baseline_right_hand_pressure.sh
bash scripts/PressureSensor/knife/train_baseline_right_no_wrench.sh
bash scripts/PressureSensor/knife/train_baseline_right_no_wrench_pressure.sh
