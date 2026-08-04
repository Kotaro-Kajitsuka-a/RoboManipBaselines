from os import path

import mujoco
import numpy as np
from gymnasium.spaces import Box, Dict

from robo_manip_baselines.common import DataKey

from .MujocoXarm7EnvBase import MujocoXarm7EnvBase
from .MujocoXarm7PushtEnv import MujocoXarm7PushtEnv


class MujocoXarm7PushtTEnvBase(MujocoXarm7PushtEnv):
    observation_space = Dict(
        {
            **MujocoXarm7PushtEnv.observation_space.spaces,
            "tblock_pose": Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float64),
        }
    )
    xml_filename = None
    world_idx_range = None

    def __init__(self, **kwargs):
        assert self.xml_filename is not None
        assert self.world_idx_range is not None

        MujocoXarm7EnvBase.__init__(
            self,
            path.join(
                path.dirname(__file__),
                "../../assets/mujoco/envs/xarm7",
                self.xml_filename,
            ),
            np.array([0.0, 0.0, 0.0, 0.8, 0.0, 0.8, 0.0, *[0.0] * 6]),
            **kwargs,
        )
        self.original_tblock_pos = self.model.body("tblock").pos.copy()

    @property
    def measured_keys_to_save(self):
        return [
            *super().measured_keys_to_save,
            DataKey.MEASURED_TBLOCK_POSE,
        ]

    def _get_obs(self):
        obs = super()._get_obs()
        obs["tblock_pose"] = self.get_body_pose("tblock")
        return obs

    def get_tblock_pose_from_obs(self, obs):
        return obs["tblock_pose"]

    def modify_world(self, world_idx=None, cumulative_idx=None):
        assert world_idx is not None
        assert world_idx in self.world_idx_range, world_idx
        rng = np.random.Generator(np.random.PCG64(world_idx))

        tblock_pos = self.original_tblock_pos.copy()
        tblock_pos[:2] += rng.uniform(
            low=[-0.05, -0.09],
            high=[0.05, 0.12],
        )
        tblock_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "tblock_freejoint"
        )
        tblock_qpos_addr = self.model.jnt_qposadr[tblock_joint_id]
        self.init_qpos[tblock_qpos_addr : tblock_qpos_addr + 3] = tblock_pos
        yaw = rng.uniform(low=-np.pi, high=np.pi)
        self.init_qpos[tblock_qpos_addr + 3 : tblock_qpos_addr + 7] = np.array(
            [np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)]
        )

        return world_idx
