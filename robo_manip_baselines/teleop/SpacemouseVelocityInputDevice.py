import numpy as np
import pinocchio as pin

from .SpacemouseInputDevice import SpacemouseInputDevice


class SpacemouseVelocityInputDevice(SpacemouseInputDevice):
    """Measured-position targets with incremental SpaceMouse rotation."""

    max_position_offset = 0.02  # [m]

    def set_command_data(self):
        measured_eef_pos = self.arm_manager.get_eef_pose_from_joint_pos(
            np.asarray(
                self.arm_manager.env.unwrapped.arm_joint_pos_actual, dtype=np.float64
            )
        )[:3]
        position_offset = self.pos_scale * np.array(
            [-self.state.y, self.state.x, self.state.z]
        )
        offset_norm = np.linalg.norm(position_offset)
        if offset_norm > self.max_position_offset:
            position_offset *= self.max_position_offset / offset_norm

        delta_rpy = self.rpy_scale * np.array(
            [-self.state.roll, -self.state.pitch, -2.0 * self.state.yaw]
        )

        # Re-anchor translation to the measurement and accumulate rotation input.
        target_se3 = self.arm_manager.target_se3.copy()
        target_se3.translation = measured_eef_pos + position_offset
        target_se3.rotation = pin.rpy.rpyToMatrix(*delta_rpy) @ target_se3.rotation
        self.arm_manager.set_command_eef_pose(target_se3)

        gripper_joint_pos = self.arm_manager.get_command_gripper_joint_pos().copy()
        if self.state.buttons[0] > 0 and self.state.buttons[-1] <= 0:
            gripper_joint_pos += self.gripper_scale
        elif self.state.buttons[-1] > 0 and self.state.buttons[0] <= 0:
            gripper_joint_pos -= self.gripper_scale
        self.arm_manager.set_command_gripper_joint_pos(gripper_joint_pos)
