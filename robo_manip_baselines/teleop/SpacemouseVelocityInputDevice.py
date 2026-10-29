import numpy as np

from .SpacemouseInputDevice import SpacemouseInputDevice


class SpacemouseVelocityInputDevice(SpacemouseInputDevice):
    """Translation-only position targets relative to the measured EEF position."""

    max_position_offset = 0.01  # [m]

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

        # Re-anchor every command to the measurement; keep the commanded rotation.
        target_se3 = self.arm_manager.target_se3.copy()
        target_se3.translation = measured_eef_pos + position_offset
        self.arm_manager.set_command_eef_pose(target_se3)

        gripper_joint_pos = self.arm_manager.get_command_gripper_joint_pos().copy()
        if self.state.buttons[0] > 0 and self.state.buttons[-1] <= 0:
            gripper_joint_pos += self.gripper_scale
        elif self.state.buttons[-1] > 0 and self.state.buttons[0] <= 0:
            gripper_joint_pos -= self.gripper_scale
        self.arm_manager.set_command_gripper_joint_pos(gripper_joint_pos)
