import time

import numpy as np
import pinocchio as pin

from robo_manip_baselines.common import ArmManager, get_se3_from_pose

from .RealXarm7DemoEnv import RealXarm7DemoEnv


class SafeXarm7ArmManager(ArmManager):
    def set_command_eef_pose(self, eef_pose):
        if isinstance(eef_pose, pin.SE3):
            target_se3 = eef_pose.copy()
        else:
            target_se3 = get_se3_from_pose(eef_pose)

        # Use measured joints, not current_se3 (which follows commanded joints).
        measured_eef_pos = self.get_eef_pose_from_joint_pos(
            np.asarray(self.env.unwrapped.arm_joint_pos_actual, dtype=np.float64)
        )[:3]
        position_offset = target_se3.translation - measured_eef_pos
        position_offset_norm = np.linalg.norm(position_offset)
        max_position_offset = 1e-3 * self.env.unwrapped.max_eef_position_offset
        if position_offset_norm > max_position_offset:
            target_se3.translation = measured_eef_pos + (
                max_position_offset * position_offset / position_offset_norm
            )

        super().set_command_eef_pose(target_se3)


class RealXarm7SafeDemoEnv(RealXarm7DemoEnv):
    min_eef_z = -125.0  # [mm]
    # Maximum distance from measured EEF position to commanded EEF position [mm].
    max_eef_position_offset = 50.0
    reset_eef_speed = 30  # 10 mm/sec when robot reset
    reset_eef_position = [
        325.564056,
        -1.945497,
        124.241219,
        179.208765,
        -0.0106,
        0.809589,
    ]  # default value  x=400, y=-50, z=150, roll=-180, pitch=0, yaw=0,

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.body_config_list[0].BodyManagerClass = SafeXarm7ArmManager
        self._check_xarm_code(
            "set_tcp_offset",
            self.xarm_api.set_tcp_offset([0.0, 0.0, 300.0, 0.0, 0.0, 0.0]),
        )

    def get_input_device_kwargs(self, input_device_name):
        if input_device_name == "spacemouse":
            return {"pos_scale": 4e-3, "rpy_scale": 5e-3, "gripper_scale": 20.0}
        elif input_device_name == "spacemouse_vel":
            return {"pos_scale": 2e-2, "gripper_scale": 20.0}
        else:
            return super().get_input_device_kwargs(input_device_name)

    def _reset_robot(self):
        time.sleep(0.2)
        self.xarm_api.set_mode(7)
        self.xarm_api.set_state(0)
        time.sleep(0.2)

        start_time = time.time()
        while True:
            xarm_code, measured_eef_pose = self.xarm_api.get_position(is_radian=False)
            self._check_xarm_code("get_position", xarm_code)
            position_reached = all(
                abs(measured_eef_pose[i] - self.reset_eef_position[i]) < 10.0  # 10.0 mm
                for i in range(3)
            )
            # Wrap angular differences to [-180, 180) degrees.
            orientation_errors = [
                (measured_eef_pose[i] - self.reset_eef_position[i] + 180.0) % 360.0
                - 180.0
                for i in range(3, 6)
            ]
            orientation_reached = all(
                abs(error) < 1.0 for error in orientation_errors
            )  # 1.0deg
            if position_reached and orientation_reached:
                break

            if time.time() - start_time >= 10.0:
                stop_code = self.xarm_api.set_state(4)
                position_errors = [
                    measured_eef_pose[i] - self.reset_eef_position[i] for i in range(3)
                ]
                raise TimeoutError(
                    f"Reset EEF pose timeout after 10 seconds. "
                    f"Position errors (mm): {position_errors}, "
                    f"Orientation errors (degrees): {orientation_errors}, "
                    f"Measured pose: {measured_eef_pose}, "
                    f"Target pose: {self.reset_eef_position}; "
                    f"set_state(4) code={stop_code}"
                )

            self._get_obs(measured_eef_pose)  # for safety check
            xarm_code = self.xarm_api.set_position(
                *self.reset_eef_position,
                speed=self.reset_eef_speed,
                is_radian=False,
                wait=False,
            )
            self._check_xarm_code("set_position", xarm_code)
            time.sleep(0.005)

        time.sleep(0.2)
        self.xarm_api.set_mode(6)
        self.xarm_api.set_state(0)
        time.sleep(4.0)

        super()._reset_robot()

    def _get_obs(self, measured_eef_pose=None):
        obs = super()._get_obs()

        if measured_eef_pose is None:
            xarm_code, measured_eef_pose = self.xarm_api.get_position(is_radian=False)
            self._check_xarm_code("get_position", xarm_code)
        measured_eef_z = measured_eef_pose[2]  # [mm]
        if measured_eef_z < self.min_eef_z:
            stop_code = self.xarm_api.set_state(4)
            raise RuntimeError(
                f"[{self.__class__.__name__}] Measured EEF z is below the lower limit: "
                f"measured={measured_eef_z:.6f} mm, limit={self.min_eef_z:.6f} mm; "
                f"set_state(4) code={stop_code}"
            )

        return obs
