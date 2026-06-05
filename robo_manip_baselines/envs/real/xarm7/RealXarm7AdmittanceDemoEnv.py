import time

import numpy as np

from .RealXarm7EnvBase import RealXarm7EnvBase


class RealXarm7AdmittanceEnvBase(RealXarm7EnvBase):
    def __init__(
        self, robot_ip, camera_ids, gelsight_ids=None, init_qpos=None, **kwargs
    ):
        super().__init__(
            self,
            robot_ip,
            camera_ids,
            gelsight_ids,
            init_qpos,
            **kwargs,
        )

        # set tool admittance parameters:
        K_pos = 300  #  x/y/z linear stiffness coefficient, range: 0 ~ 2000 (N/m)
        K_ori = 4  #  Rx/Ry/Rz rotational stiffness coefficient, range: 0 ~ 20 (Nm/rad)

        # Attention: for M and J, smaller value means less effort to drive the arm, but may also be less stable, please be careful.
        M = float(0.06)  #  x/y/z equivalent mass; range: 0.02 ~ 1 kg
        J = (
            M * 0.01
        )  #  Rx/Ry/Rz equivalent moment of inertia, range: 1e-4 ~ 0.01 (Kg*m^2)

        c_axis = [1, 1, 1, 1, 1, 1]  # set z axis as compliant axis
        ref_frame = 0  # 0 : base , 1 : tool

        self.xarm_api.set_ft_sensor_admittance_parameters(
            [M, M, M, J, J, J], [K_pos, K_pos, K_pos, K_ori, K_ori, K_ori], [0] * 6
        )  # B(damping) is reserved, give zeros
        self.xarm_api.set_ft_sensor_admittance_parameters(ref_frame, c_axis)
        self.xarm_api.set_ft_sensor_enable(1)
        # will overwrite previous sensor zero and payload configuration
        # self.xarm_api.set_ft_sensor_zero() # remove this if zero_offset and payload already identified & compensated!
        # Todo: in the super().__init__(), there is a line of self.xarm_api.set_ft_sensor_zero(). It might be better to remove it to avoid overwriting the admittance control configuration.
        time.sleep(0.2)  # wait for writing zero operation to take effect, do not remove

        self.xarm_api.clean_error()
        self.xarm_api.set_mode(6)
        self.xarm_api.set_state(0)
        # move robot in admittance control application
        print(self.xarm_api.get_ft_sensor_mode())
        self.xarm_api.set_ft_sensor_mode(1)
        # will start after set_state(0)
        self.xarm_api.set_state(0)

        time.sleep(100000)
        self.xarm_api.set_ft_sensor_mode(0)
        self.xarm_api.set_ft_sensor_enable(0)
        self.xarm_api.disconnect()


class RealXarm7AdmittanceDemoEnv(RealXarm7AdmittanceEnvBase):
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
