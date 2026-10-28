import time

from .RealXarm7DemoEnv import RealXarm7DemoEnv


class RealXarm7SafeDemoEnv(RealXarm7DemoEnv):
    min_eef_z = 40.0  # [mm]
    reset_eef_speed = 30  # 10 mm/sec when robot reset
    reset_eef_position = [
        325.564056,
        -1.945497,
        124.241219,
        179.208765,
        -0.0106,
        0.809589,
    ]  # default value  x=400, y=-50, z=150, roll=-180, pitch=0, yaw=0,

    def _reset_robot(self):
        time.sleep(0.2)
        self.xarm_api.set_mode(7)
        self.xarm_api.set_state(0)
        time.sleep(0.2)

        while True:
            xarm_code, measured_eef_pose = self.xarm_api.get_position(is_radian=False)
            self._check_xarm_code("get_position", xarm_code)
            position_reached = all(
                abs(measured_eef_pose[i] - self.reset_eef_position[i]) < 1.0
                for i in range(3)
            )
            # Wrap angular differences to [-180, 180) degrees.
            orientation_errors = [
                (measured_eef_pose[i] - self.reset_eef_position[i] + 180.0) % 360.0
                - 180.0
                for i in range(3, 6)
            ]
            orientation_reached = all(abs(error) < 1.0 for error in orientation_errors)
            if position_reached and orientation_reached:
                break
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
