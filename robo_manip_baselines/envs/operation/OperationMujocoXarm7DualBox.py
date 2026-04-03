import gymnasium as gym


class OperationMujocoXarm7DualBox:
    def setup_env(self, render_mode="human"):
        self.env = gym.make(
            "robo_manip_baselines/MujocoXarm7DualBoxEnv-v0",
            render_mode=render_mode,
        )

    def get_pre_motion_phases(self):
        return []
