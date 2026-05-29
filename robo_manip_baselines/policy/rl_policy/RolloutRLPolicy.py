from robo_manip_baselines.policy.ppo_cus import RolloutPpoCus


class RolloutRLPolicy(RolloutPpoCus):
    """
    Rollout entry point for RL policies trained outside RoboManipBaselines.

    The first supported backend is the ManiSkill-style PPO checkpoint path that
    RolloutPpoCus already runs. Keeping this as a separate policy name lets us
    evolve the RL rollout path without adding more task-specific behavior to
    RolloutPpoCus.
    """

    pass
