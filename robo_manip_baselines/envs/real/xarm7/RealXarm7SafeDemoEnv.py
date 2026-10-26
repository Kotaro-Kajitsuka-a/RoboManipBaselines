import time
from typing import override

import numpy as np

from .RealXarm7EnvBase import RealXarm7EnvBase


class RealXarm7DemoEnv(RealXarm7EnvBase):
    reset_eef_speed = 10  # 10 mm/sec when robot reset
    reset_eef_position = [
        325.564056,
        -1.945497,
        124.241219,
        179.208765,
        -0.0106,
        0.809589,
    ]  # default value  x=400, y=-50, z=150, roll=-180, pitch=0, yaw=0,

    def __init__(
        self,
        **kwargs,
    ):
        RealXarm7EnvBase.__init__(
            self,
            init_qpos=np.concatenate(
                [np.deg2rad([0.0, -30.0, 0.0, 45.0, 0.0, 75.0, 0.0]), np.array([800.0])]
            ),
            **kwargs,
        )

    def get_input_device_kwargs(self, input_device_name):
        if input_device_name == "spacemouse":
            return {"gripper_scale": 20.0}
        else:
            return super().get_input_device_kwargs(input_device_name)

    def modify_world(self, world_idx=None, cumulative_idx=None):
        """Modify simulation world depending on world index."""
        # TODO: Automatically set world index according to task variations
        if world_idx is None:
            world_idx = 0
            # world_idx = cumulative_idx % 2
        return world_idx

    @override
    def _reset_robot(self):
        time.sleep(0.2)
        self.xarm_api.set_mode(7)
        self.set_state(0)
        time.sleep(0.2)
        self.xarm_api.set_position(
            x=self.reset_eef_position[0],
            y=self.reset_eef_position[1],
            z=self.reset_eef_position[2],
            roll=self.reset_eef_position[3],
            pitch=self.reset_eef_position[4],
            yaw=self.reset_eef_position[5],
            speed=self.reset_eef_speed,
            is_radian=False,
            wait=True,
        )

        super()._reset_robot()

        time.sleep(0.2)
        self.xarm_api.set_mode(6)
        self.set_state(0)
        time.sleep(0.2)
