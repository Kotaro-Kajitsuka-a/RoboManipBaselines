import pinocchio as pin

from robo_manip_baselines.common import ArmManager, get_se3_from_pose

from .RealXarm7FixedGripperDemoEnv import RealXarm7FixedGripperDemoEnv


class PlanarConstraintArmManager(ArmManager):
    def set_command_eef_pose(self, eef_pose):
        if isinstance(eef_pose, pin.SE3):
            target_se3 = eef_pose.copy()
        else:
            target_se3 = get_se3_from_pose(eef_pose)

        target_se3.translation[2] = self._original_target_se3.translation[2]
        target_se3.rotation = self._original_target_se3.rotation.copy()

        super().set_command_eef_pose(target_se3)


class RealXarm7PlanarConstraintDemoEnv(RealXarm7FixedGripperDemoEnv):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.body_config_list[0].BodyManagerClass = PlanarConstraintArmManager
