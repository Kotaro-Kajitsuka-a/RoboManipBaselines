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
    reset_joint_pos_tolerance = np.deg2rad(0.1)  # [rad]

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.body_config_list[0].BodyManagerClass = PlanarConstraintArmManager

        arm_config = self.body_config_list[0]
        self.eef_pin_model = pin.buildModelFromUrdf(arm_config.arm_urdf_path)
        self.eef_pin_data = self.eef_pin_model.createData()
        self.original_eef_z = self._get_eef_z(arm_config.init_arm_joint_pos)

    def _reset_robot(self):
        print(
            f"[{self.__class__.__name__}] Start moving the robot to the reset position."
        )

        arm_config = self.body_config_list[0]
        obs = self._get_obs()
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

    def _get_obs(self):
        obs = super()._get_obs()

        arm_config = self.body_config_list[0]
        arm_joint_pos = obs["joint_pos"][arm_config.arm_joint_idxes]
        measured_eef_z = self._get_eef_z(arm_joint_pos)
        if measured_eef_z < self.original_eef_z - self.max_eef_z_drop:
            raise RuntimeError(
                f"[{self.__class__.__name__}] Measured EEF z is below the planar constraint: "
                f"original={self.original_eef_z:.6f} m, measured={measured_eef_z:.6f} m"
            )

        return obs

    def _get_eef_z(self, arm_joint_pos):
        pin.forwardKinematics(
            self.eef_pin_model,
            self.eef_pin_data,
            arm_joint_pos,
        )
        return self.eef_pin_data.oMi[
            self.body_config_list[0].ik_eef_joint_id
        ].translation[2]
