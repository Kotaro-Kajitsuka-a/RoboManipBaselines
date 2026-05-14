from os import path

import mujoco
import numpy as np

from .MujocoXarm7AdmittanceEnvBase import MujocoXarm7AdmittanceEnvBase
from .MujocoXarm7PushtEnv import MujocoXarm7PushtEnv


class MujocoXarm7AdmittancePusht3Env(
    MujocoXarm7AdmittanceEnvBase, MujocoXarm7PushtEnv
):
    metadata = MujocoXarm7AdmittanceEnvBase.metadata.copy()

    def __init__(self, **kwargs):
        MujocoXarm7AdmittanceEnvBase.__init__(
            self,
            path.join(
                path.dirname(__file__),
                "../../assets/mujoco/envs/xarm7/env_xarm7_pusht3.xml",
            ),
            np.array([0.0, 0.0, 0.0, 0.8, 0.0, 0.8, 0.0, *[0.0] * 6]),
            **kwargs,
        )
        self.original_tblock_pos = self.model.body("tblock").pos.copy()
        self.tblock_pos_offsets = np.array(
            [
                [0.0, -0.06, 0.0],
                [0.0, -0.03, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.03, 0.0],
                [0.0, 0.06, 0.0],
                [0.0, 0.09, 0.0],
            ]
        )

    def modify_world(self, world_idx=None, cumulative_idx=None):
        if world_idx is None:
            world_idx = 0
        assert world_idx == 0, world_idx

        tblock_pos = self.original_tblock_pos.copy()
        tblock_pos[:2] += np.random.uniform(
            low=[-0.05, -0.09],
            high=[0.05, 0.12],
        )
        tblock_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "tblock_freejoint"
        )
        tblock_qpos_addr = self.model.jnt_qposadr[tblock_joint_id]
        self.init_qpos[tblock_qpos_addr : tblock_qpos_addr + 3] = tblock_pos
        yaw = np.random.uniform(low=-np.pi, high=np.pi)
        self.init_qpos[tblock_qpos_addr + 3 : tblock_qpos_addr + 7] = np.array(
            [np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)]
        )

        return world_idx
