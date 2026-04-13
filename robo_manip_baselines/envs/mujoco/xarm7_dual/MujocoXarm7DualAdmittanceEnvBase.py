import numpy as np

from robo_manip_baselines.common import DataKey, EefAdmittanceArmManager

from ..MujocoMultiRateEnvBase import MujocoMultiRateEnvBase
from .MujocoXarm7DualEnvBase import MujocoXarm7DualEnvBase


class MujocoXarm7DualAdmittanceEnvBase(MujocoMultiRateEnvBase, MujocoXarm7DualEnvBase):
    def __init__(self, xml_file, init_qpos, **kwargs):
        super().__init__(xml_file, init_qpos, **kwargs)
        substep_wrench_shape = DataKey.get_dim(
            DataKey.SUBSTEP_MEASURED_EEF_WRENCH, self
        )
        self._substep_measured_eef_wrench_seq = np.zeros(
            substep_wrench_shape, dtype=np.float64
        )
        self._substep_compensated_eef_wrench_seq = np.zeros(
            substep_wrench_shape, dtype=np.float64
        )
        self._substep_compensated_lpf_eef_wrench_seq = np.zeros(
            substep_wrench_shape, dtype=np.float64
        )
        self._substep_measured_eef_wrench_idx = 0
        self._base_target_eef_pose = np.zeros(
            DataKey.get_dim(DataKey.COMMAND_EEF_POSE, self), dtype=np.float64
        )
        self._base_target_gripper_joint_pos = np.zeros(
            DataKey.get_dim(DataKey.COMMAND_GRIPPER_JOINT_POS, self), dtype=np.float64
        )
        self._current_ctrl = self.init_qpos[: self.model.nu].copy()
        self._admittance_arm_manager_list = [
            EefAdmittanceArmManager(
                self,
                body_config,
                arm_name=arm_name,
                gravity_comp_body_names=self._get_gravity_comp_body_names(arm_name),
            )
            for body_config, arm_name in zip(self.body_config_list, ("left", "right"))
        ]

    @property
    def command_keys_for_step(self):
        return [DataKey.COMMAND_EEF_POSE, DataKey.COMMAND_GRIPPER_JOINT_POS]

    @property
    def command_keys_to_save(self):
        return [DataKey.COMMAND_EEF_POSE, DataKey.COMMAND_GRIPPER_JOINT_POS]

    def reset_model(self):
        obs = super().reset_model()
        self._reset_multirate_state(obs)
        return obs

    def get_substep_wrench_data_to_save(self):
        return {
            DataKey.SUBSTEP_MEASURED_EEF_WRENCH: self._substep_measured_eef_wrench_seq.copy(),
            DataKey.SUBSTEP_COMPENSATED_EEF_WRENCH: self._substep_compensated_eef_wrench_seq.copy(),
            DataKey.SUBSTEP_COMPENSATED_LPF_EEF_WRENCH: self._substep_compensated_lpf_eef_wrench_seq.copy(),
        }

    def _on_policy_step(self, action):
        action = np.asarray(action, dtype=np.float64)
        self._substep_measured_eef_wrench_seq.fill(0.0)
        self._substep_compensated_eef_wrench_seq.fill(0.0)
        self._substep_compensated_lpf_eef_wrench_seq.fill(0.0)
        self._substep_measured_eef_wrench_idx = 0
        pose_dim = DataKey.get_dim(DataKey.COMMAND_EEF_POSE, self)
        self._base_target_eef_pose = action[:pose_dim].copy()
        self._base_target_gripper_joint_pos = action[pose_dim:].copy()
        for body_manager in self._admittance_arm_manager_list:
            eef_idx = body_manager.body_config.eef_idx
            body_manager.set_command_eef_pose(
                self._base_target_eef_pose[7 * eef_idx : 7 * (eef_idx + 1)]
            )
            body_manager.set_command_gripper_joint_pos(
                self._base_target_gripper_joint_pos[
                    body_manager.body_config.gripper_joint_idxes_in_gripper_joint_pos
                ]
            )

    def _on_admittance_step(self):
        obs = self._get_obs()
        measured_joint_pos = self.get_joint_pos_from_obs(obs)
        measured_wrench = self.get_eef_wrench_from_obs(obs)
        self._substep_measured_eef_wrench_seq[self._substep_measured_eef_wrench_idx] = (
            measured_wrench.copy()
        )
        for body_manager in self._admittance_arm_manager_list:
            eef_idx = body_manager.body_config.eef_idx
            body_manager.sync_with_measurement(
                measured_joint_pos[body_manager.body_config.arm_joint_idxes],
                measured_joint_pos[body_manager.body_config.gripper_joint_idxes],
                measured_wrench[6 * eef_idx : 6 * (eef_idx + 1)],
            )
            step_result = body_manager.update_command(self.admittance_timestep)
            self._substep_compensated_eef_wrench_seq[
                self._substep_measured_eef_wrench_idx, 6 * eef_idx : 6 * (eef_idx + 1)
            ] = step_result.comp_wrench.copy()
            self._substep_compensated_lpf_eef_wrench_seq[
                self._substep_measured_eef_wrench_idx, 6 * eef_idx : 6 * (eef_idx + 1)
            ] = step_result.comp_wrench_lpf.copy()
            arm_joint_pos, gripper_joint_pos = body_manager.get_command_joint_pos()
            self._current_ctrl[body_manager.body_config.arm_joint_idxes] = arm_joint_pos
            self._current_ctrl[body_manager.body_config.gripper_joint_idxes] = (
                gripper_joint_pos
            )
        self._substep_measured_eef_wrench_idx += 1

    def _apply_ctrl(self):
        self.data.ctrl[:] = self._current_ctrl

    def _reset_multirate_state(self, obs):
        measured_joint_pos = self.get_joint_pos_from_obs(obs)
        measured_wrench = self.get_eef_wrench_from_obs(obs)
        self._substep_measured_eef_wrench_seq.fill(0.0)
        self._substep_compensated_eef_wrench_seq.fill(0.0)
        self._substep_compensated_lpf_eef_wrench_seq.fill(0.0)
        self._substep_measured_eef_wrench_idx = 0
        for body_manager in self._admittance_arm_manager_list:
            eef_idx = body_manager.body_config.eef_idx
            body_manager.reset()
            body_manager.sync_with_measurement(
                measured_joint_pos[body_manager.body_config.arm_joint_idxes],
                measured_joint_pos[body_manager.body_config.gripper_joint_idxes],
                measured_wrench[6 * eef_idx : 6 * (eef_idx + 1)],
            )
            body_manager.set_command_joint_pos(
                measured_joint_pos[body_manager.body_config.arm_joint_idxes],
                measured_joint_pos[body_manager.body_config.gripper_joint_idxes],
            )
            arm_joint_pos, gripper_joint_pos = body_manager.get_command_joint_pos()
            self._current_ctrl[body_manager.body_config.arm_joint_idxes] = arm_joint_pos
            self._current_ctrl[body_manager.body_config.gripper_joint_idxes] = (
                gripper_joint_pos
            )

        self._base_target_eef_pose = np.concatenate(
            [
                body_manager.get_command_eef_pose()
                for body_manager in self._admittance_arm_manager_list
            ]
        )
        self._base_target_gripper_joint_pos = np.concatenate(
            [
                body_manager.get_command_gripper_joint_pos()
                for body_manager in self._admittance_arm_manager_list
            ]
        )

    @staticmethod
    def _get_gravity_comp_body_names(arm_name):
        return [
            f"{arm_name}/xarm_gripper_base_link",
            f"{arm_name}/left_outer_knuckle",
            f"{arm_name}/left_inner_knuckle",
            f"{arm_name}/left_finger",
            f"{arm_name}/right_outer_knuckle",
            f"{arm_name}/right_inner_knuckle",
            f"{arm_name}/right_finger",
        ]
