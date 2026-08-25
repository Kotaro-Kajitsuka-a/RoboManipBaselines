import numpy as np
import pinocchio as pin

from robo_manip_baselines.common import ArmManager, get_se3_from_pose

from .RealXarm7FixedGripperDemoEnv import RealXarm7FixedGripperDemoEnv


class PlanarConstraintArmManager(ArmManager):
    max_eef_speed = 0.02  # [m/s]

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
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.body_config_list[0].BodyManagerClass = PlanarConstraintArmManager
