import gymnasium as gym

from robo_manip_baselines.common import GraspPhaseBase


class GraspPhase(GraspPhaseBase):
    def set_target(self):
        self.set_target_open()


class OperationMujocoXarm7Pusht:
    def setup_env(self, render_mode="human"):
        self.env = gym.make(
            "robo_manip_baselines/MujocoXarm7PushtEnv-v0", render_mode=render_mode
        )

    def get_pre_motion_phases(self):
        return [
            GraspPhase(self),
        ]
