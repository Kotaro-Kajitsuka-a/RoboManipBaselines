import numpy as np
from gymnasium.spaces import Box, Dict

from robo_manip_baselines.common import DataKey, EefAdmittanceArmManager

from ..MujocoMultiRateEnvBase import MujocoMultiRateEnvBase
from .MujocoXarm7EnvBase import MujocoXarm7EnvBase


class MujocoXarm7AdmittanceEnvBase(MujocoMultiRateEnvBase, MujocoXarm7EnvBase):
    observation_space = Dict(
        {
            "joint_pos": Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float64),
            "joint_vel": Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float64),
            "wrench": Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float64),
            "wrench_moving_average": Box(
                low=-np.inf, high=np.inf, shape=(6,), dtype=np.float64
            ),
        }
    )

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
        self._admittance_arm_manager = EefAdmittanceArmManager(
            self,
            self.body_config_list[0],
            arm_name=None,
            gravity_comp_body_names=self._get_gravity_comp_body_names(),
        )

    @property
    def command_keys_for_step(self):
        return [DataKey.COMMAND_EEF_POSE, DataKey.COMMAND_GRIPPER_JOINT_POS]

    @property
    def command_keys_to_save(self):
        return [DataKey.COMMAND_EEF_POSE, DataKey.COMMAND_GRIPPER_JOINT_POS]

    @property
    def measured_keys_to_save(self):
        return [
            DataKey.MEASURED_JOINT_POS,
            DataKey.MEASURED_JOINT_VEL,
            DataKey.MEASURED_GRIPPER_JOINT_POS,
            DataKey.MEASURED_EEF_POSE,
            DataKey.MEASURED_EEF_WRENCH,
            DataKey.MEASURED_EEF_WRENCH_MOVING_AVERAGE,
        ]

    def reset_model(self):
        obs = super().reset_model()
        self._reset_multirate_state(obs)
        return obs

    def _get_obs(self):
        obs = super()._get_obs()
        obs["wrench_moving_average"] = self._get_wrench_moving_average()
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
        self._admittance_arm_manager.set_command_eef_pose(self._base_target_eef_pose)
        self._admittance_arm_manager.set_command_gripper_joint_pos(
            self._base_target_gripper_joint_pos
        )

    def _on_admittance_step(self):
        obs = self._get_obs()
        measured_joint_pos = self.get_joint_pos_from_obs(obs)
        measured_wrench = self.get_eef_wrench_from_obs(obs)
        # measured_wrench[0] += 25.0
        self._substep_measured_eef_wrench_seq[self._substep_measured_eef_wrench_idx] = (
            measured_wrench.copy()
        )
        self._admittance_arm_manager.sync_with_measurement(
            measured_joint_pos[self.body_config_list[0].arm_joint_idxes],
            measured_joint_pos[self.body_config_list[0].gripper_joint_idxes],
            measured_wrench,
        )
        step_result = self._admittance_arm_manager.update_command(
            self.admittance_timestep
        )
        self._substep_compensated_eef_wrench_seq[
            self._substep_measured_eef_wrench_idx
        ] = step_result.comp_wrench.copy()
        self._substep_compensated_lpf_eef_wrench_seq[
            self._substep_measured_eef_wrench_idx
        ] = step_result.comp_wrench_lpf.copy()
        self._substep_measured_eef_wrench_idx += 1
        arm_joint_pos, gripper_joint_pos = (
            self._admittance_arm_manager.get_command_joint_pos()
        )
        self._current_ctrl[:7] = arm_joint_pos
        self._current_ctrl[7] = gripper_joint_pos[0]

    def _apply_ctrl(self):
        self.data.ctrl[:] = self._current_ctrl

    def _get_wrench_moving_average(self):
        if self._substep_measured_eef_wrench_idx == 0:
            return np.zeros(6, dtype=np.float64)

        return np.mean(
            self._substep_measured_eef_wrench_seq[
                : self._substep_measured_eef_wrench_idx
            ],
            axis=0,
            dtype=np.float64,
        )

    def _reset_multirate_state(self, obs):
        measured_joint_pos = self.get_joint_pos_from_obs(obs)
        measured_wrench = self.get_eef_wrench_from_obs(obs)
        self._substep_measured_eef_wrench_seq.fill(0.0)
        self._substep_compensated_eef_wrench_seq.fill(0.0)
        self._substep_compensated_lpf_eef_wrench_seq.fill(0.0)
        self._substep_measured_eef_wrench_idx = 0
        self._admittance_arm_manager.reset()
        self._admittance_arm_manager.sync_with_measurement(
            measured_joint_pos[self.body_config_list[0].arm_joint_idxes],
            measured_joint_pos[self.body_config_list[0].gripper_joint_idxes],
            measured_wrench,
        )
        self._admittance_arm_manager.set_command_joint_pos(
            measured_joint_pos[self.body_config_list[0].arm_joint_idxes],
            measured_joint_pos[self.body_config_list[0].gripper_joint_idxes],
        )
        arm_joint_pos, gripper_joint_pos = (
            self._admittance_arm_manager.get_command_joint_pos()
        )
        self._current_ctrl[:7] = arm_joint_pos
        self._current_ctrl[7] = gripper_joint_pos[0]
        self._base_target_eef_pose = self._admittance_arm_manager.get_command_eef_pose()
        self._base_target_gripper_joint_pos = (
            self._admittance_arm_manager.get_command_gripper_joint_pos().copy()
        )

    @staticmethod
    def _get_gravity_comp_body_names():
        return [
            "xarm_gripper_base_link",
            "left_outer_knuckle",
            "left_inner_knuckle",
            "left_finger",
            "right_outer_knuckle",
            "right_inner_knuckle",
            "right_finger",
        ]
