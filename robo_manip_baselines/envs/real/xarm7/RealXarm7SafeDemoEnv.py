import time
from typing import override

from .RealXarm7DemoEnv import RealXarm7DemoEnv


class RealXarm7DemoEnv(RealXarm7DemoEnv):
    reset_eef_speed = 10  # 10 mm/sec when robot reset
    reset_eef_position = [
        325.564056,
        -1.945497,
        124.241219,
        179.208765,
        -0.0106,
        0.809589,
    ]  # default value  x=400, y=-50, z=150, roll=-180, pitch=0, yaw=0,

    @override
    def _reset_robot(self):
        time.sleep(0.2)
        self.xarm_api.set_mode(7)
        self.set_state(0)
        time.sleep(0.2)
        self.xarm_api.set_position(
            *self.reset_eef_position,
            speed=self.reset_eef_speed,
            is_radian=False,
            wait=True,
        )

        super()._reset_robot()

        time.sleep(0.2)
        self.xarm_api.set_mode(6)
        self.set_state(0)
        time.sleep(0.2)
