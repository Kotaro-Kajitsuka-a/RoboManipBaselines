import gymnasium as gym
import numpy as np
import pinocchio as pin

from robo_manip_baselines.common import GraspPhaseBase, ReachPhaseBase


def get_target_se3(op, pos_z):
    target_pos = op.env.unwrapped.get_body_pose("cable_end")[0:3]
    target_pos[2] = pos_z
    return pin.SE3(np.diag([-1, -1, 1]), target_pos)


class ReachPhase1(ReachPhaseBase):
    def set_target(self):
        self.target_se3 = get_target_se3(
            self.op,
            pos_z=1.08,  # [m]
        )
        self.duration = 0.7  # [s]


class ReachPhase2(ReachPhaseBase):
    def set_target(self):
        self.target_se3 = get_target_se3(
            self.op,
            pos_z=1.04,  # [m]
        )
        self.duration = 0.3  # [s]


class GraspPhase(GraspPhaseBase):
    def set_target(self):
        self.set_target_close()


class OperationMujocoKinovaGen3Cable:
    def setup_env(self, render_mode="human"):
        self.env = gym.make(
            "robo_manip_baselines/MujocoKinovaGen3CableEnv-v0", render_mode=render_mode
        )

    def get_pre_motion_phases(self):
        return [
            ReachPhase1(self),
            ReachPhase2(self),
            GraspPhase(self),
        ]
