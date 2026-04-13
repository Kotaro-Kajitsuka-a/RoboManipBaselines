from os import path

import numpy as np

from .MujocoXarm7DualEnvBase import MujocoXarm7DualEnvBase
from .Xarm7DualBoxWorldMixin import Xarm7DualBoxWorldMixin


class MujocoXarm7DualBoxEnv(Xarm7DualBoxWorldMixin, MujocoXarm7DualEnvBase):
    def __init__(
        self,
        **kwargs,
    ):
        init_arm_joint_pos = np.deg2rad([0.0, -30.0, 0.0, 45.0, 0.0, 75.0, 0.0])
        init_gripper_joint_pos = np.zeros(6)

        MujocoXarm7DualEnvBase.__init__(
            self,
            path.join(
                path.dirname(__file__),
                "../../assets/mujoco/envs/xarm7_dual/env_xarm7_dual_box.xml",
            ),
            np.concatenate(
                [
                    init_arm_joint_pos,
                    init_gripper_joint_pos,
                    init_arm_joint_pos,
                    init_gripper_joint_pos,
                ]
            ),
            **kwargs,
        )
        self._setup_box_world_params()
