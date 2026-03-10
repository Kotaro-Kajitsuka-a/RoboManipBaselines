import gymnasium as gym
import numpy as np
import pinocchio as pin

from robo_manip_baselines.common import GraspPhaseBase, ReachPhaseBase


def get_target_se3(op, pos_z):
    target_pos = op.env.unwrapped.get_body_pose("cable_end")[0:3]
    target_pos[1] -= 0.005
    target_pos[2] = pos_z
    return pin.SE3(pin.rpy.rpyToMatrix(3.12986747, 0.01612632, 0.73595797), target_pos)


class ReachPhase1(ReachPhaseBase):
    def set_target(self):
        self.target_se3 = get_target_se3(
            self.op,
            pos_z=1.15,  # [m]
        )
        self.duration = 1.0  # [s]


class ReachPhase2(ReachPhaseBase):
    def set_target(self):
        self.target_se3 = get_target_se3(
            self.op,
            pos_z=1.058,  # [m]
        )
        self.duration = 0.5  # [s]


class GraspPhase(GraspPhaseBase):
    def set_target(self):
        self.gripper_joint_pos = np.array([0])
        self.duration = 0.2


class OperationMujocoPandaCable:
    def setup_env(self, render_mode="human"):
        self.env = gym.make(
            "robo_manip_baselines/MujocoPandaCableEnv-v0", render_mode=render_mode
        )

    def get_pre_motion_phases(self):
        return [
            ReachPhase1(self),
            ReachPhase2(self),
            GraspPhase(self),
        ]
