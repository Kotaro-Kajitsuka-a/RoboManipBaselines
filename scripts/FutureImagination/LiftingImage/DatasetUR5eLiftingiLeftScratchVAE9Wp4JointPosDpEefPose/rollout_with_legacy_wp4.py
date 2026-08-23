from legacy_wp4_compat import load_policy

from robo_manip_baselines.policy.wrench_predictor4_online import (
    WrenchPredictor4OnlineUtils,
)

WrenchPredictor4OnlineUtils.load_policy = load_policy


if __name__ == "__main__":
    from robo_manip_baselines.bin.Rollout import RolloutMain

    main = RolloutMain()
    main.run()
