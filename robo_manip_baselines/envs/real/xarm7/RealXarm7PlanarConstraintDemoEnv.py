import numpy as np
import pinocchio as pin

from robo_manip_baselines.common import ArmManager, get_se3_from_pose

from .RealXarm7FixedGripperDemoEnv import RealXarm7FixedGripperDemoEnv


class PlanarConstraintArmManager(ArmManager):
    max_eef_speed = 0.10  # [m/s]

    def set_command_joint_pos(self, arm_joint_pos, gripper_joint_pos):
        raise RuntimeError(
            f"[{self.__class__.__name__}] Joint position commands bypass the planar constraint. Use command_eef_pose or command_eef_pose_rel."
        )

    def set_command_eef_pose(self, eef_pose):
        if isinstance(eef_pose, pin.SE3):
            next_target_se3 = eef_pose.copy()
        else:
            next_target_se3 = get_se3_from_pose(eef_pose)

        next_target_se3.translation[2] = self._original_target_se3.translation[2]
        next_target_se3.rotation = self._original_target_se3.rotation.copy()

        translation_delta = next_target_se3.translation - self.target_se3.translation
        translation_distance = np.linalg.norm(translation_delta)
        max_translation_distance = self.max_eef_speed * self.env.unwrapped.dt
        if translation_distance > max_translation_distance:
            next_target_se3.translation[:] = self.target_se3.translation + (
                max_translation_distance * translation_delta / translation_distance
            )

        super().set_command_eef_pose(next_target_se3)


class RealXarm7PlanarConstraintDemoEnv(RealXarm7FixedGripperDemoEnv):
    max_eef_z_drop = 0.005  # [m]
    reset_approach_z_offset = 0.05  # [m]
    reset_eef_pos_tolerance = 0.01  # [m]
    reset_joint_pos_tolerance = np.deg2rad(0.1)  # [rad]

    fixed_gripper_joint_pos = np.array([0.0], dtype=np.float64)
    fixed_gripper_init_qpos = np.concatenate(
        [
            np.deg2rad([1.1, 17.8, -1.4, 23.5, -2.4, 5.6, 1.3]),
            fixed_gripper_joint_pos,
        ]
    )

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.body_config_list[0].BodyManagerClass = PlanarConstraintArmManager

        arm_config = self.body_config_list[0]
        self.reset_arm_manager = ArmManager(self, arm_config)
        self.init_eef_se3 = self.reset_arm_manager.current_se3.copy()

        self.xarm_api.set_collision_sensitivity(5)

    def _reset_robot(self):
        print(
            f"[{self.__class__.__name__}] Start moving the robot to the reset position."
        )

        arm_config = self.body_config_list[0]
        obs = self._get_obs()
        arm_joint_pos = obs["joint_pos"][arm_config.arm_joint_idxes]
        arm_joint_pos_error = arm_config.init_arm_joint_pos - arm_joint_pos
        if np.max(np.abs(arm_joint_pos_error)) > self.reset_joint_pos_tolerance:
            obs = self._move_eef_to_reset_approach_pose(obs)

        while True:
            arm_joint_pos = obs["joint_pos"][arm_config.arm_joint_idxes]
            arm_joint_pos_error = arm_config.init_arm_joint_pos - arm_joint_pos
            if np.max(np.abs(arm_joint_pos_error)) <= self.reset_joint_pos_tolerance:
                break

            self._set_action(
                self.init_qpos,
                duration=self.dt,
                joint_vel_limit_scale=0.1,
                wait=True,
            )
            obs = self._get_obs()

        print(
            f"[{self.__class__.__name__}] Finish moving the robot to the reset position."
        )

    def _move_eef_to_reset_approach_pose(self, obs):
        print(
            f"[{self.__class__.__name__}] Start moving the EEF to the reset approach pose."
        )

        arm_config = self.body_config_list[0]
        approach_se3 = self.init_eef_se3.copy()
        approach_se3.translation[2] += self.reset_approach_z_offset

        while True:
            eef_pos_error = (
                approach_se3.translation
                - self.reset_arm_manager.current_se3.translation
            )
            if np.linalg.norm(eef_pos_error) <= self.reset_eef_pos_tolerance:
                break

            self.reset_arm_manager.set_command_eef_pose(approach_se3)
            action = self.init_qpos.copy()
            action[arm_config.arm_joint_idxes] = self.reset_arm_manager.arm_joint_pos
            self._set_action(
                action,
                duration=self.dt,
                joint_vel_limit_scale=0.1,
                wait=True,
            )
            obs = self._get_obs()

        print(
            f"[{self.__class__.__name__}] Finish moving the EEF to the reset approach pose."
        )
        return obs

    def _reset_robot(self):
        print(
            f"[{self.__class__.__name__}] Start moving the robot to the reset position."
        )

        arm_config = self.body_config_list[0]
        obs = self._get_obs()
        arm_joint_pos = obs["joint_pos"][arm_config.arm_joint_idxes]
        arm_joint_pos_error = arm_config.init_arm_joint_pos - arm_joint_pos
        if np.max(np.abs(arm_joint_pos_error)) > self.reset_joint_pos_tolerance:
            obs = self._move_eef_to_reset_approach_pose()

        while True:
            arm_joint_pos = obs["joint_pos"][arm_config.arm_joint_idxes]
            arm_joint_pos_error = arm_config.init_arm_joint_pos - arm_joint_pos
            if np.max(np.abs(arm_joint_pos_error)) <= self.reset_joint_pos_tolerance:
                break

            self._set_action(
                self.init_qpos,
                duration=self.dt,
                joint_vel_limit_scale=0.1,
                wait=True,
            )
            obs = self._get_obs()

        print(
            f"[{self.__class__.__name__}] Finish moving the robot to the reset position."
        )

    def _move_eef_to_reset_approach_pose(self):
        print(
            f"[{self.__class__.__name__}] Start moving the EEF to the reset approach pose."
        )

        arm_config = self.body_config_list[0]
        approach_se3 = self.init_eef_se3.copy()
        approach_se3.translation[2] += self.reset_approach_z_offset
        self.reset_arm_manager.set_command_eef_pose(approach_se3)

        action = self.init_qpos.copy()
        action[arm_config.arm_joint_idxes] = self.reset_arm_manager.arm_joint_pos
        self._set_action(
            action,
            duration=2.0,
            joint_vel_limit_scale=0.1,
            wait=True,
        )
        obs = self._get_obs()

        print(
            f"[{self.__class__.__name__}] Finish moving the EEF to the reset approach pose."
        )
        return obs

    def _get_obs(self):
        obs = super()._get_obs()

        arm_config = self.body_config_list[0]
        joint_pos = obs["joint_pos"]
        self.reset_arm_manager.set_command_joint_pos(
            joint_pos[arm_config.arm_joint_idxes],
            joint_pos[arm_config.gripper_joint_idxes],
        )
        measured_eef_z = self.reset_arm_manager.current_se3.translation[2]
        init_eef_z = self.init_eef_se3.translation[2]
        if measured_eef_z < init_eef_z - self.max_eef_z_drop:
            raise RuntimeError(
                f"[{self.__class__.__name__}] Measured EEF z is below the planar constraint: "
                f"init={init_eef_z:.6f} m, measured={measured_eef_z:.6f} m"
            )

        return obs
