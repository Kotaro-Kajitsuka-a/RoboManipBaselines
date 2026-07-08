from os import path

import numpy as np

from robo_manip_baselines.common import ArmConfig

from .RealXarm7DualEnvBase import RealXarm7DualEnvBase


class RealXarm7DualFixedGripperDemoEnv(RealXarm7DualEnvBase):
    fixed_gripper_joint_pos = np.array([119.0, 119.0], dtype=np.float64)

    def __init__(
        self,
        **kwargs,
    ):
        RealXarm7DualEnvBase.__init__(
            self,
            # Left robot arm index:0, Right robot arm index:1 . This is the same system as UR5eDual.
            init_qpos=np.concatenate(
                [
                    np.deg2rad([0.0, -30.0, 0.0, 45.0, 0.0, 75.0, 0.0]),
                    self.fixed_gripper_joint_pos[[0]],
                    np.deg2rad([0.0, -30.0, 0.0, 45.0, 0.0, 75.0, 0.0]),
                    self.fixed_gripper_joint_pos[[1]],
                ]
            ),
            **kwargs,
        )

        self.body_config_list = [
            ArmConfig(
                arm_urdf_path=path.join(
                    path.dirname(__file__),
                    "../../assets/common/robots/xarm7/xarm7.urdf",
                ),
                arm_root_pose=None,
                ik_eef_joint_id=7,
                arm_joint_idxes=np.arange(7),
                gripper_joint_idxes=np.array([7]),
                gripper_joint_idxes_in_gripper_joint_pos=np.array([0]),
                eef_idx=0,
                init_arm_joint_pos=self.init_qpos[0:7],
                init_gripper_joint_pos=self.fixed_gripper_joint_pos[[0]],
            ),
            ArmConfig(
                arm_urdf_path=path.join(
                    path.dirname(__file__),
                    "../../assets/common/robots/xarm7/xarm7.urdf",
                ),
                arm_root_pose=None,
                ik_eef_joint_id=7,
                arm_joint_idxes=np.arange(8, 15),
                gripper_joint_idxes=np.array([15]),
                gripper_joint_idxes_in_gripper_joint_pos=np.array([1]),
                eef_idx=1,
                init_arm_joint_pos=self.init_qpos[8:15],
                init_gripper_joint_pos=self.fixed_gripper_joint_pos[[1]],
            ),
        ]

    def _set_action(self, action, duration=None, joint_vel_limit_scale=0.5, wait=False):
        action = action.copy()
        action[self.body_config_list[0].gripper_joint_idxes] = (
            self.fixed_gripper_joint_pos[0]
        )
        action[self.body_config_list[1].gripper_joint_idxes] = (
            self.fixed_gripper_joint_pos[1]
        )
        super()._set_action(action, duration, joint_vel_limit_scale, wait)

    def get_input_device_kwargs(self, input_device_name):
        if input_device_name == "spacemouse":
            return {
                0: {"gripper_scale": 20.0},
                1: {"gripper_scale": 20.0},
            }
        else:
            return super().get_input_device_kwargs(input_device_name)

    def modify_world(self, world_idx=None, cumulative_idx=None):
        """Modify simulation world depending on world index."""
        # TODO: Automatically set world index according to task variations
        if world_idx is None:
            world_idx = 0
            # world_idx = cumulative_idx % 2
        return world_idx
