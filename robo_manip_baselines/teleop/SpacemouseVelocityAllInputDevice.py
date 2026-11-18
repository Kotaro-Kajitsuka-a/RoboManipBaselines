import numpy as np
import pinocchio as pin

from robo_manip_baselines.common import get_se3_from_pose

from .SpacemouseInputDevice import SpacemouseInputDevice


class SpacemouseVelocityAllInputDevice(SpacemouseInputDevice):
    """SpaceMouse offsets relative to measured EEF position and orientation."""

    def set_command_data(self):
        measured_eef_pose = self.arm_manager.get_eef_pose_from_joint_pos(
            np.asarray(
                self.arm_manager.env.unwrapped.arm_joint_pos_actual, dtype=np.float64
            )
        )
        target_se3 = get_se3_from_pose(measured_eef_pose)
        target_se3.translation += self.pos_scale * np.array(
            [-self.state.y, self.state.x, self.state.z]
        )
        delta_rpy = self.rpy_scale * np.array(
            [-self.state.roll, -self.state.pitch, -2.0 * self.state.yaw]
        )
        target_se3.rotation = pin.rpy.rpyToMatrix(*delta_rpy) @ target_se3.rotation
        self.arm_manager.set_command_eef_pose(target_se3)

        gripper_joint_pos = self.arm_manager.get_command_gripper_joint_pos().copy()
        if self.state.buttons[0] > 0 and self.state.buttons[-1] <= 0:
            gripper_joint_pos += self.gripper_scale
        elif self.state.buttons[-1] > 0 and self.state.buttons[0] <= 0:
            gripper_joint_pos -= self.gripper_scale
        self.arm_manager.set_command_gripper_joint_pos(gripper_joint_pos)
