from os import path

import mujoco
import numpy as np

from .MujocoXarm7DualEnvBase import MujocoXarm7DualEnvBase


class MujocoXarm7DualBoxEnv(MujocoXarm7DualEnvBase):
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
        self.box_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "box_geom"
        )
        self.box_white_mat_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_MATERIAL, "box_white"
        )
        self.box_red_mat_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_MATERIAL, "box_red"
        )
        self.box_green_mat_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_MATERIAL, "box_green"
        )
        self.box_orange_mat_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_MATERIAL, "box_orange"
        )
        self.box_pale_yellow_mat_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_MATERIAL, "box_pale_yellow"
        )
        self.box_pale_blue_mat_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_MATERIAL, "box_pale_blue"
        )
        self.box_pale_pink_mat_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_MATERIAL, "box_pale_pink"
        )
        self.box_pale_mint_mat_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_MATERIAL, "box_pale_mint"
        )
        self.box_pale_lavender_mat_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_MATERIAL, "box_pale_lavender"
        )
        self.box_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "box"
        )
        self.box_base_mass = float(self.model.body_mass[self.box_body_id])
        self.box_base_inertia = self.model.body_inertia[self.box_body_id].copy()
        self.box_base_ipos = self.model.body_ipos[self.box_body_id].copy()
        self.box_half_size = self.model.geom_size[self.box_geom_id].copy()

        self.box_center_com_offset = self._com_offset_from_ratio()
        self.box_high_com_offset = self._com_offset_from_ratio(z_ratio=100.0)
        self.box_low_com_offset = self._com_offset_from_ratio(z_ratio=-100.0)
        self.box_xarm_side_com_offset = self._com_offset_from_ratio(x_ratio=-50.0)
        self.box_xarm_side_more_com_offset = self._com_offset_from_ratio(x_ratio=-100.0)

    def _com_offset_from_ratio(self, x_ratio=0.0, y_ratio=0.0, z_ratio=0.0):
        ratio = np.array([x_ratio, y_ratio, z_ratio], dtype=np.float64)
        assert np.all(
            np.abs(ratio) <= 100.0
        ), f"[{self.__class__.__name__}] CoM ratio must be within [-100, 100], got {ratio}"
        return self.box_half_size * (ratio / 100.0)

    def modify_world(self, world_idx=None, cumulative_idx=None):
        if world_idx is None:
            world_idx = 0

        assert world_idx in (
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
        ), f"[{self.__class__.__name__}] Invalid world_idx: {world_idx}"

        if world_idx == 0:
            box_mat_id = self.box_white_mat_id
            mass_scale = 1.0
            com_offset = self.box_center_com_offset
        elif world_idx == 1:
            box_mat_id = self.box_red_mat_id
            mass_scale = 0.5
            com_offset = self.box_center_com_offset
        elif world_idx == 2:
            box_mat_id = self.box_green_mat_id
            mass_scale = 1.5
            com_offset = self.box_center_com_offset
        elif world_idx == 3:
            box_mat_id = self.box_orange_mat_id
            mass_scale = 0.1
            com_offset = self.box_center_com_offset
        elif world_idx == 4:
            box_mat_id = self.box_pale_yellow_mat_id
            mass_scale = 1.0
            com_offset = self.box_high_com_offset
        elif world_idx == 5:
            box_mat_id = self.box_pale_blue_mat_id
            mass_scale = 1.0
            com_offset = self.box_low_com_offset
        elif world_idx == 6:
            box_mat_id = self.box_pale_pink_mat_id
            mass_scale = 1.0
            com_offset = self.box_xarm_side_com_offset
        elif world_idx == 7:
            box_mat_id = self.box_pale_mint_mat_id
            mass_scale = 1.0
            com_offset = self.box_xarm_side_more_com_offset + self.box_high_com_offset
        else:
            box_mat_id = self.box_pale_lavender_mat_id
            mass_scale = 1.0
            com_offset = self.box_xarm_side_more_com_offset + self.box_low_com_offset

        self.model.geom_matid[self.box_geom_id] = box_mat_id
        self.model.body_mass[self.box_body_id] = self.box_base_mass * mass_scale
        self.model.body_inertia[self.box_body_id] = self.box_base_inertia * mass_scale
        self.model.body_ipos[self.box_body_id] = self.box_base_ipos + com_offset

        return world_idx

    @property
    def camera_names(self):
        return ["front"]
