import numpy as np
import pinocchio as pin

from .InputDeviceBase import InputDeviceBase


class ViveInputDevice(InputDeviceBase):
    """HTC Vive Controller for teleoperation input device."""

    def __init__(
        self,
        arm_manager,
        pos_scale=1.0,
        gripper_scale=5.0,
        device_params=None,
        vive_to_eef_frame_rotation=None,
    ):
        super().__init__()
        assert device_params is not None

        self.arm_manager = arm_manager
        self.pos_scale = pos_scale
        self.gripper_scale = gripper_scale
        self.name = device_params["name"]
        self.serial_number = device_params["serial_number"]
        if vive_to_eef_frame_rotation is None:
            self.vive_to_eef_frame_rotation = np.eye(3)
        else:
            self.vive_to_eef_frame_rotation = np.array(
                vive_to_eef_frame_rotation, dtype=np.float64
            )
        assert self.vive_to_eef_frame_rotation.shape == (3, 3)

    def connect(self):
        self.enabled_teleop = False
        self.prev_is_enable_teleop_pressed = False
        self.vive_se3_at_enable = None
        self.eef_se3_at_enable = None
        self.has_announced_ready = False
        if self.connected:
            return

        import openvr

        self.vr_system = openvr.init(openvr.VRApplication_Other)
        self.openvr = openvr
        self.connected = True

    def close(self):
        if self.connected:
            self.openvr.shutdown()
            self.connected = False

    def read(self):
        if not self.connected:
            raise RuntimeError(f"[{self.__class__.__name__}] Device is not connected.")

        self._read_vive()
        if self.name not in self.state:
            self.prev_is_enable_teleop_pressed = False
            self.has_announced_ready = False
            return

        # Use the trigger axis (axes[2]) to enable teleop.
        is_enable_teleop_pressed = self.state[self.name]["axes"][2] == 1.0
        if is_enable_teleop_pressed and not self.prev_is_enable_teleop_pressed:
            self.enabled_teleop = True
            self.vive_se3_at_enable = self.state[self.name]["se3"].copy()
            self.eef_se3_at_enable = self.arm_manager.current_se3.copy()
            print(f"[{self.__class__.__name__}] Enable teleop.")
        self.prev_is_enable_teleop_pressed = is_enable_teleop_pressed

    def _read_vive(self):
        openvr = self.openvr
        poses = self.vr_system.getDeviceToAbsoluteTrackingPose(
            openvr.TrackingUniverseStanding, 0, openvr.k_unMaxTrackedDeviceCount
        )

        state = {}

        for i in range(openvr.k_unMaxTrackedDeviceCount):
            pose = poses[i]

            if not pose.bDeviceIsConnected or not pose.bPoseIsValid:
                continue

            device_class = self.vr_system.getTrackedDeviceClass(i)
            if device_class != openvr.TrackedDeviceClass_Controller:
                continue

            sn = self.vr_system.getStringTrackedDeviceProperty(
                i, openvr.Prop_SerialNumber_String
            )

            if sn != self.serial_number:
                continue

            mat = np.eye(4)
            mat[:3, :4] = pose.mDeviceToAbsoluteTracking.m

            se3 = pin.SE3(mat[:3, :3], mat[:3, 3])

            _, vive_state = self.vr_system.getControllerState(i)
            axes = np.array(
                [
                    vive_state.rAxis[0].x,
                    vive_state.rAxis[0].y,
                    vive_state.rAxis[1].x,
                ]
            )
            buttons = {
                "application_menu": bool(
                    vive_state.ulButtonPressed >> openvr.k_EButton_ApplicationMenu & 1
                ),
                "trackpad_pressed": bool(
                    vive_state.ulButtonPressed >> openvr.k_EButton_SteamVR_Touchpad & 1
                ),
            }

            state[self.name] = {
                "se3": se3,
                "axes": axes,
                "buttons": buttons,
            }

        self.state = state
        if (self.name in state) and (not self.has_announced_ready):
            print(
                f"[{self.__class__.__name__}] Vive is ready. Pull the trigger fully to enable teleop."
            )
            self.has_announced_ready = True

    def is_ready(self):
        return (self.state is not None) and (self.name in self.state)

    def set_command_data(self):
        if (not self.enabled_teleop) or (self.name not in self.state):
            return

        delta_vive_se3 = (
            self.vive_se3_at_enable.inverse() * self.state[self.name]["se3"]
        )
        adjusted_delta_vive_se3 = pin.SE3(
            self.vive_to_eef_frame_rotation
            @ delta_vive_se3.rotation
            @ self.vive_to_eef_frame_rotation.T,
            self.vive_to_eef_frame_rotation @ delta_vive_se3.translation,
        )
        scaled_delta_vive_se3 = pin.SE3(
            adjusted_delta_vive_se3.rotation,
            self.pos_scale * adjusted_delta_vive_se3.translation,
        )
        target_se3 = self.eef_se3_at_enable * scaled_delta_vive_se3

        self.arm_manager.set_command_eef_pose(target_se3)

        # Set gripper command
        gripper_joint_pos = self.arm_manager.get_command_gripper_joint_pos().copy()
        buttons = self.state[self.name]["buttons"]
        if buttons["application_menu"] and not buttons["trackpad_pressed"]:
            gripper_joint_pos -= self.gripper_scale
        elif buttons["trackpad_pressed"] and not buttons["application_menu"]:
            gripper_joint_pos += self.gripper_scale

        self.arm_manager.set_command_gripper_joint_pos(gripper_joint_pos)
