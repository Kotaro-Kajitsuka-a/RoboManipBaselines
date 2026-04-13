import mujoco
import numpy as np


class Xarm7DualBoxWorldMixin:
    def _setup_box_world_params(self):
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
        self.box_xarm_side_com_offset = self._com_offset_from_ratio(x_ratio=-50.0)
        self.box_anti_xarm_side_com_offset = self._com_offset_from_ratio(x_ratio=50.0)
        self.box_left_hand_side_com_offset = self._com_offset_from_ratio(y_ratio=50.0)
        self.box_right_hand_side_com_offset = self._com_offset_from_ratio(y_ratio=-50.0)

        self._box_variant_params = {
            0: (self.box_white_mat_id, 1.0, self.box_center_com_offset),
            1: (self.box_red_mat_id, 2.0, self.box_center_com_offset),
            2: (self.box_green_mat_id, 4.0, self.box_center_com_offset),
            3: (self.box_orange_mat_id, 0.5, self.box_center_com_offset),
            4: (self.box_pale_yellow_mat_id, 0.1, self.box_center_com_offset),
            5: (self.box_pale_blue_mat_id, 1.0, self.box_xarm_side_com_offset),
            6: (self.box_pale_pink_mat_id, 1.0, self.box_anti_xarm_side_com_offset),
            7: (self.box_pale_mint_mat_id, 1.0, self.box_left_hand_side_com_offset),
            8: (
                self.box_pale_lavender_mat_id,
                1.0,
                self.box_right_hand_side_com_offset,
            ),
        }

    def _com_offset_from_ratio(self, x_ratio=0.0, y_ratio=0.0, z_ratio=0.0):
        ratio = np.array([x_ratio, y_ratio, z_ratio], dtype=np.float64)
        assert np.all(
            np.abs(ratio) <= 100.0
        ), f"[{self.__class__.__name__}] CoM ratio must be within [-100, 100], got {ratio}"
        return self.box_half_size * (ratio / 100.0)

    def modify_world(self, world_idx=None, cumulative_idx=None):
        if world_idx is None:
            world_idx = 0

        assert (
            world_idx in self._box_variant_params
        ), f"[{self.__class__.__name__}] Invalid world_idx: {world_idx}"

        box_mat_id, mass_scale, com_offset = self._box_variant_params[world_idx]

        self.model.geom_matid[self.box_geom_id] = box_mat_id
        self.model.body_mass[self.box_body_id] = self.box_base_mass * mass_scale
        self.model.body_inertia[self.box_body_id] = self.box_base_inertia * mass_scale
        self.model.body_ipos[self.box_body_id] = self.box_base_ipos + com_offset

        return world_idx

    @property
    def camera_names(self):
        return ["front"]
