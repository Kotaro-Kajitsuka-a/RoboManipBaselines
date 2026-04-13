import threading

import numpy as np

from .InputDeviceBase import InputDeviceBase


class KeyboardInputDevice(InputDeviceBase):
    """Keyboard for teleoperation input device."""

    def __init__(
        self,
        arm_manager,
        pos_scale=1e-2,
        rpy_scale=5e-2,
        gripper_scale=50.0,
    ):
        super().__init__()

        self.arm_manager = arm_manager
        self.pos_scale = pos_scale
        self.rpy_scale = rpy_scale
        self.gripper_scale = gripper_scale

        self.state = {
            # position control keys
            "w": False,
            "s": False,
            "a": False,
            "d": False,
            "q": False,
            "e": False,
            #  rotation control keys
            "i": False,
            "k": False,
            "j": False,
            "l": False,
            "u": False,
            "o": False,
            # gripper control keys
            "z": False,  # close gripper
            "x": False,  # open gripper
        }

        self.listener = None
        self.listener_thread = None

    def connect(self):
        if self.connected:
            return

        from pynput import keyboard

        self.listener = keyboard.Listener(
            on_press=self._on_press, on_release=self._on_release
        )

        # keyboard listener another thread
        self.listener_thread = threading.Thread(target=self._start_listener)
        self.listener_thread.daemon = True
        self.listener_thread.start()

        self.connected = True
        print(f"[{self.__class__.__name__}] Connected.")
        print(f"""[{self.__class__.__name__}] Key Bindings:
  - WASD : XY movement
  - QE   : Z-axis movement
  - IJKL : Roll and Pitch rotation
  - UO  : Yaw rotation
  - Z/X  : Gripper open/close""")

    def _start_listener(self):
        self.listener.start()
        self.listener.join()

    def _on_press(self, key):
        try:
            k = key.char.lower()
            if k in self.state:
                self.state[k] = True
        except AttributeError:
            pass

    def _on_release(self, key):
        try:
            k = key.char.lower()
            if k in self.state:
                self.state[k] = False
        except AttributeError:
            pass

    def read(self):
        if not self.connected:
            raise RuntimeError(f"[{self.__class__.__name__}] Device is not connected.")

    def set_command_data(self):
        target_se3 = self.arm_manager.target_se3.copy()

        current_z = target_se3.translation[2]
        print(current_z)
        target_z = current_z - 0.0005
        threshhold_z = 0.90
        if target_z < threshhold_z:
            target_z = threshhold_z

        target_se3.translation = np.array([0.208189, 3.00184e-06, 1.0199730490988117])
        target_se3.rotation = np.array(
            [
                [1, 5.26999e-06, 5.87127e-12],
                [5.26999e-06, -1, 5.11829e-06],
                [3.28442e-11, -5.11829e-06, -1],
            ]
        )

        self.arm_manager.set_command_eef_pose(target_se3)

        # Set gripper command
        gripper_joint_pos = self.arm_manager.get_command_gripper_joint_pos().copy()

        if self.state["z"] and not self.state["x"]:
            gripper_joint_pos += self.gripper_scale
        elif self.state["x"] and not self.state["z"]:
            gripper_joint_pos -= self.gripper_scale

        self.arm_manager.set_command_gripper_joint_pos(gripper_joint_pos)

    def disconnect(self):
        if self.connected:
            if self.listener:
                self.listener.stop()
            self.connected = False
            print(f"[{self.__class__.__name__}] Disconnected.")
