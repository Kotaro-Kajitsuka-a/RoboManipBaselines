from dataclasses import dataclass

import mujoco
import numpy as np
import pinocchio as pin

from ..utils.MathUtils import get_pose_from_se3, get_se3_from_pose
from .ArmManager import ArmManager


@dataclass
class EefAdmittanceStepResult:
    raw_wrench: np.ndarray
    comp_wrench: np.ndarray
    comp_wrench_lpf: np.ndarray
    compliant_target_se3: pin.SE3


class SimplifiedImpedanceController:
    """mc_rtc-style impedance with fd=0, pd_dot=0, pd_ddot=0."""

    def __init__(self):
        self.wrench_lpf_cutoff_period = 0.004
        self.target_wrench = np.zeros(6, dtype=np.float64)

        # mc_rtc sample values
        self.M = np.array([10.0, 10.0, 10.0, 2.0, 2.0, 2.0], dtype=np.float64)
        self.D = np.array(
            [1500.0, 1500.0, 1500.0, 200.0, 200.0, 200.0], dtype=np.float64
        )
        self.K = np.array(
            [1500.0, 1500.0, 200.0, 200.0, 200.0, 200.0], dtype=np.float64
        )
        self.Kf = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)

        # Default values kept here for quick temporary rollback.
        # self.M = np.array([2.0, 2.0, 2.0, 0.2, 0.2, 0.2], dtype=np.float64)
        # self.D = np.array([80.0, 80.0, 80.0, 8.0, 8.0, 8.0], dtype=np.float64)
        # self.K = np.array([300.0, 300.0, 300.0, 30.0, 30.0, 30.0], dtype=np.float64)
        # self.Kf = np.ones(6, dtype=np.float64)

        self.delta_pose_lin_limit = 0.05
        self.delta_pose_ang_limit = 0.2
        self.delta_vel_lin_limit = 0.2
        self.delta_vel_ang_limit = 1.0
        self.delta_acc_lin_limit = 20.0
        self.delta_acc_ang_limit = 20.0

        self.reset()

    def reset(self):
        self.wrench_comp_lpf = np.zeros(6, dtype=np.float64)
        self.delta_pose_se3_world = pin.SE3.Identity()
        self.delta_vel_world = np.zeros(6, dtype=np.float64)
        self.delta_acc_world = np.zeros(6, dtype=np.float64)

    def update(self, base_target_se3, measured_se3, comp_wrench, dt):
        comp_wrench = np.asarray(comp_wrench, dtype=np.float64)
        lpf_period = max(self.wrench_lpf_cutoff_period, 2.0 * dt)
        x = 1.0 if lpf_period <= dt else dt / lpf_period
        self.wrench_comp_lpf = (
            x * comp_wrench + (1.0 - x) * self.wrench_comp_lpf
        ).astype(np.float64)

        control_rot_world = measured_se3.rotation
        delta_pose_world = self._get_delta_pose_world()
        delta_pose_surface = self._transform_world_vec_to_local(
            control_rot_world, delta_pose_world
        )
        delta_vel_surface = self._transform_world_vec_to_local(
            control_rot_world, self.delta_vel_world
        )

        delta_acc_surface = (
            -self.D * delta_vel_surface
            - self.K * delta_pose_surface
            + self.Kf * (self.wrench_comp_lpf - self.target_wrench)
        ) / self.M
        delta_acc_surface[:3] = self._clip_vec_norm(
            delta_acc_surface[:3], self.delta_acc_lin_limit
        )
        delta_acc_surface[3:] = self._clip_vec_norm(
            delta_acc_surface[3:], self.delta_acc_ang_limit
        )

        self.delta_acc_world = self._transform_local_vec_to_world(
            control_rot_world, delta_acc_surface
        )
        self.delta_vel_world = self.delta_vel_world + dt * self.delta_acc_world
        self.delta_vel_world[:3] = self._clip_vec_norm(
            self.delta_vel_world[:3], self.delta_vel_lin_limit
        )
        self.delta_vel_world[3:] = self._clip_vec_norm(
            self.delta_vel_world[3:], self.delta_vel_ang_limit
        )

        delta_motion_world = dt * (
            self.delta_vel_world + 0.5 * dt * self.delta_acc_world
        )
        self._integrate_delta_pose(delta_motion_world)
        self._clip_delta_pose()

        return EefAdmittanceStepResult(
            raw_wrench=np.zeros(6, dtype=np.float64),
            comp_wrench=comp_wrench.copy(),
            comp_wrench_lpf=self.wrench_comp_lpf.copy(),
            compliant_target_se3=self._calc_compliant_target_se3(base_target_se3),
        )

    def _calc_compliant_target_se3(self, base_target_se3):
        target_rot_only = pin.SE3(base_target_se3.rotation, np.zeros(3))
        return (
            target_rot_only
            * self.delta_pose_se3_world
            * target_rot_only.inverse()
            * base_target_se3
        )

    def _get_delta_pose_world(self):
        return np.concatenate(
            [
                self.delta_pose_se3_world.translation.copy(),
                pin.log3(self.delta_pose_se3_world.rotation),
            ]
        )

    def _integrate_delta_pose(self, delta_motion_world):
        next_translation = (
            self.delta_pose_se3_world.translation + delta_motion_world[:3]
        )
        next_rotation = (
            pin.exp3(delta_motion_world[3:]) @ self.delta_pose_se3_world.rotation
        )
        self.delta_pose_se3_world = pin.SE3(next_rotation, next_translation)

    def _clip_delta_pose(self):
        translation = self._clip_vec_norm(
            self.delta_pose_se3_world.translation.copy(), self.delta_pose_lin_limit
        )
        rotvec = self._clip_vec_norm(
            pin.log3(self.delta_pose_se3_world.rotation), self.delta_pose_ang_limit
        )
        self.delta_pose_se3_world = pin.SE3(pin.exp3(rotvec), translation)

    @staticmethod
    def _transform_world_vec_to_local(rot_world, world_vec):
        return np.concatenate(
            [rot_world.T @ world_vec[:3], rot_world.T @ world_vec[3:]]
        )

    @staticmethod
    def _transform_local_vec_to_world(rot_world, local_vec):
        return np.concatenate([rot_world @ local_vec[:3], rot_world @ local_vec[3:]])

    @staticmethod
    def _clip_vec_norm(vec, max_norm):
        norm = float(np.linalg.norm(vec))
        if norm > max_norm and norm > 1e-12:
            return vec * (max_norm / norm)
        return vec


class EefAdmittanceArmManager(ArmManager):
    """
    Admittance-specific arm manager used inside multi-rate admittance environments.

    This class reuses Pinocchio-based FK/IK utilities from `ArmManager`, but it is not
    intended to be used as a generic body manager in `MotionManager`.
    In particular, end-effector commands are stored as base targets first, then
    converted into compliant targets after admittance updates.
    """

    def __init__(self, env, body_config, arm_name, gravity_comp_body_names):
        self.arm_name = arm_name
        self.gravity_comp_body_names = gravity_comp_body_names
        self.controller = SimplifiedImpedanceController()
        super().__init__(env, body_config)

    def reset(self, init=False):
        super().reset(init=init)
        self.controller.reset()
        self.measured_arm_joint_pos = self.arm_joint_pos.copy()
        self.measured_gripper_joint_pos = self.gripper_joint_pos.copy()
        self.measured_se3 = self.current_se3.copy()
        self.measured_wrench = np.zeros(6, dtype=np.float64)
        self.target_se3 = self._original_target_se3.copy()
        self.compliant_target_se3 = self.target_se3.copy()
        self.gravity_comp_com_world = self.current_se3.translation.copy()

    def sync_with_measurement(
        self, measured_joint_pos, measured_gripper_joint_pos, raw_wrench
    ):
        self.measured_arm_joint_pos = np.asarray(
            measured_joint_pos, dtype=np.float64
        ).copy()
        self.measured_gripper_joint_pos = np.asarray(
            measured_gripper_joint_pos, dtype=np.float64
        ).copy()
        self.measured_wrench = np.asarray(raw_wrench, dtype=np.float64).copy()
        self.measured_se3 = get_se3_from_pose(
            self.get_eef_pose_from_joint_pos(self.measured_arm_joint_pos)
        )

    def set_command_data(self, key, command, is_skip=False):
        del key, command, is_skip
        raise NotImplementedError(
            f"[{self.__class__.__name__}] Do not use generic set_command_data(). "
            "Use the explicit admittance methods instead."
        )

    def get_command_data(self, key):
        del key
        raise NotImplementedError(
            f"[{self.__class__.__name__}] Do not use generic get_command_data(). "
            "Use explicit getters such as get_command_joint_pos() instead."
        )

    def set_command_joint_pos(self, arm_joint_pos, gripper_joint_pos):
        super().set_command_joint_pos(arm_joint_pos, gripper_joint_pos)
        self.target_se3 = self.current_se3.copy()
        self.compliant_target_se3 = self.target_se3.copy()
        self.controller.reset()

    def set_command_eef_pose(self, eef_pose):
        if isinstance(eef_pose, pin.SE3):
            self.target_se3 = pin.SE3(
                eef_pose.rotation.copy(), eef_pose.translation.copy()
            )
        else:
            self.target_se3 = get_se3_from_pose(eef_pose)

    def set_command_eef_pose_rel(self, eef_pose_rel, is_skip=False):
        if is_skip:
            return
        target_se3 = self.target_se3 * pin.SE3(
            pin.rpy.rpyToMatrix(*eef_pose_rel[3:6]), eef_pose_rel[0:3]
        )
        self.set_command_eef_pose(target_se3)

    def get_command_eef_pose(self):
        return get_pose_from_se3(self.compliant_target_se3)

    def update_command(self, dt):
        raw_wrench = self.measured_wrench.copy()
        step_result = self.controller.update(
            self.target_se3,
            self.measured_se3,
            self._calc_gravity_compensated_wrench(raw_wrench),
            dt,
        )
        step_result.raw_wrench[:] = raw_wrench
        self.compliant_target_se3 = step_result.compliant_target_se3

        self.arm_joint_pos = self.measured_arm_joint_pos.copy()
        self.forward_kinematics()
        self._solve_inverse_kinematics(self.compliant_target_se3)

        return step_result

    def draw_markers(self):
        if self.body_config.get_root_pose_func is None:
            measured_se3 = self.measured_se3
            base_target_se3 = self.target_se3
            compliant_target_se3 = self.compliant_target_se3
        else:
            root_pose_rel = (
                self.body_config.get_root_pose_func(self.env.unwrapped)
                * get_se3_from_pose(self.body_config.arm_root_pose).inverse()
            )
            measured_se3 = root_pose_rel * self.measured_se3
            base_target_se3 = root_pose_rel * self.target_se3
            compliant_target_se3 = root_pose_rel * self.compliant_target_se3

        self.env.unwrapped.draw_box_marker(
            pos=base_target_se3.translation,
            mat=base_target_se3.rotation,
            size=(0.02, 0.02, 0.03),
            rgba=(0.0, 1.0, 0.0, 0.4),
        )
        self.env.unwrapped.draw_box_marker(
            pos=compliant_target_se3.translation,
            mat=compliant_target_se3.rotation,
            size=(0.02, 0.02, 0.03),
            rgba=(0.1, 0.3, 1.0, 0.35),
        )
        self.env.unwrapped.draw_box_marker(
            pos=measured_se3.translation,
            mat=measured_se3.rotation,
            size=(0.02, 0.02, 0.03),
            rgba=(1.0, 0.0, 0.0, 0.5),
        )
        self.env.unwrapped.draw_box_marker(
            pos=self.gravity_comp_com_world,
            mat=np.eye(3),
            size=(0.01, 0.01, 0.01),
            rgba=(0.2, 1.0, 1.0, 0.8),
        )

    def _solve_inverse_kinematics(self, target_se3):
        original_target_se3 = self.target_se3
        self.target_se3 = target_se3
        self.inverse_kinematics()
        self.target_se3 = original_target_se3

    def _calc_gravity_compensated_wrench(self, raw_wrench):
        total_mass, com_world = self._calc_body_group_mass_and_com_world(
            self.gravity_comp_body_names
        )
        self.gravity_comp_com_world = com_world

        sensor_site = self.env.unwrapped.data.site(self._get_sensor_site_name())
        sensor_pos_world = sensor_site.xpos
        sensor_rot_world = sensor_site.xmat.reshape(3, 3)
        gravity_world = self.env.unwrapped.model.opt.gravity

        force_gravity_world = total_mass * gravity_world
        com_to_sensor_world = com_world - sensor_pos_world
        torque_gravity_world = np.cross(com_to_sensor_world, force_gravity_world)

        force_gravity_sensor = sensor_rot_world.T @ force_gravity_world
        torque_gravity_sensor = sensor_rot_world.T @ torque_gravity_world
        gravity_wrench_sensor = np.concatenate(
            [force_gravity_sensor, torque_gravity_sensor]
        )
        return np.asarray(raw_wrench, dtype=np.float64) + gravity_wrench_sensor

    def _calc_body_group_mass_and_com_world(self, body_name_list):
        body_id_list = []
        for body_name in body_name_list:
            body_id = mujoco.mj_name2id(
                self.env.unwrapped.model, mujoco.mjtObj.mjOBJ_BODY, body_name
            )
            assert (
                body_id >= 0
            ), f"[{self.__class__.__name__}] Unknown body name for gravity compensation: {body_name}"
            body_id_list.append(body_id)

        body_mass = self.env.unwrapped.model.body_mass[body_id_list]
        total_mass = float(body_mass.sum())
        body_com_world = self.env.unwrapped.data.xipos[body_id_list]
        com_world = (body_mass[:, None] * body_com_world).sum(axis=0) / total_mass
        return total_mass, com_world

    def _get_sensor_site_name(self):
        if self.arm_name is None:
            return "force_sensor"
        return f"{self.arm_name}/force_sensor"
