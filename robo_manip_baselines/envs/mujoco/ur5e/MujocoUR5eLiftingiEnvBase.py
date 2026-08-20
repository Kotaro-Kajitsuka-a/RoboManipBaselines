from os import path

import mujoco
import numpy as np
from gymnasium.spaces import Box, Dict

from robo_manip_baselines.common import DataKey

from .MujocoUR5eEnvBase import MujocoUR5eEnvBase


class MujocoUR5eLiftingiEnvBase(MujocoUR5eEnvBase):
    sim_timestep = 0.002
    frame_skip = 16
    success_hold_duration = 1.0  # [s]
    success_lift_threshold = 0.10  # [m]
    success_tilt_threshold = np.deg2rad(7.5)
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": int(1 / (sim_timestep * frame_skip)),
    }
    observation_space = Dict(
        {
            **MujocoUR5eEnvBase.observation_space.spaces,
            "tblock_pose": Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float64),
        }
    )
    xml_filename = None
    world_idx_range = None

    def __init__(self, allow_out_of_range_world_idx=False, **kwargs):
        assert self.xml_filename is not None
        assert self.world_idx_range is not None

        self.allow_out_of_range_world_idx = allow_out_of_range_world_idx

        MujocoUR5eEnvBase.__init__(
            self,
            path.join(
                path.dirname(__file__),
                "../../assets/mujoco/envs/ur5e",
                self.xml_filename,
            ),
            np.array(
                [
                    np.pi,
                    -np.pi / 2,
                    -0.55 * np.pi,
                    -0.45 * np.pi,
                    np.pi / 2,
                    np.pi,
                    *np.zeros(8),
                ]
            ),
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

    def _get_reward(self):
        if self._lifting_success:
            return 1.0

        tblock = self.data.body("tblock")
        lift = tblock.xpos[2] - self._initial_tblock_height

        qw, qx, qy, qz = tblock.xquat
        local_y_world_z = 2.0 * (qy * qz + qw * qx)
        tilt = np.arcsin(np.clip(np.abs(local_y_world_z), 0.0, 1.0))

        if (
            lift >= self.success_lift_threshold
            and tilt <= self.success_tilt_threshold
        ):
            self._lifting_success_duration += self.dt
        else:
            self._lifting_success_duration = 0.0

        if self._lifting_success_duration >= self.success_hold_duration:
            self._lifting_success = True

        return 1.0 if self._lifting_success else 0.0

    def reset_model(self):
        obs = super().reset_model()
        self._initial_tblock_height = self.data.body("tblock").xpos[2]
        self._lifting_success_duration = 0.0
        self._lifting_success = False
        return obs

    def modify_world(self, world_idx=None, cumulative_idx=None):
        assert world_idx is not None
        if not self.allow_out_of_range_world_idx:
            assert world_idx in self.world_idx_range, world_idx
        rng = np.random.Generator(np.random.PCG64(world_idx))

        tblock_pos = self.original_tblock_pos.copy()
        tblock_pos[:2] += rng.uniform(
            low=[-0.03, -0.05],
            high=[0.03, 0.05],
        )
        tblock_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "tblock_freejoint"
        )
        tblock_qpos_addr = self.model.jnt_qposadr[tblock_joint_id]
        self.init_qpos[tblock_qpos_addr : tblock_qpos_addr + 3] = tblock_pos
        yaw = np.pi / 2 + rng.uniform(
            low=-np.deg2rad(20),
            high=np.deg2rad(20),
        )
        self.init_qpos[tblock_qpos_addr + 3 : tblock_qpos_addr + 7] = np.array(
            [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
        )

        return world_idx
