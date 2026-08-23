from legacy_wp4_compat import load_policy

from robo_manip_baselines.policy.wrench_predictor4_online import (
    AddOnlinePbToDataset,
)

AddOnlinePbToDataset.load_policy = load_policy


if __name__ == "__main__":
    AddOnlinePbToDataset.main()
