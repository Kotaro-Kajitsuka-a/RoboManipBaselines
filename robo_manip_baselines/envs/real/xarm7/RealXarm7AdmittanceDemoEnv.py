import time

from .RealXarm7SafeDemoEnv import RealXarm7SafeDemoEnv


class RealXarm7AdmittanceDemoEnv(RealXarm7SafeDemoEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        time.sleep(0.2)
        self.setup_admittance_control()

    # override
    def get_input_device_kwargs(self, input_device_name):
        if input_device_name == "spacemouse":
            return super().get_input_device_kwargs(input_device_name)
        else:
            raise ValueError(
                f"For admittance control, only 'spasemouse' is supported. Requested config{input_device_name}"
            )

    def setup_admittance_control(self):
        # Disable admittance while configuring its parameters.
        self._check_xarm_code("set_ft_sensor_mode", self.xarm_api.set_ft_sensor_mode(0))
        # set tool admittance parameters:
        K_pos = 300  #  x/y/z linear stiffness coefficient, range: 0 ~ 2000 (N/m)
        K_ori = 4  #  Rx/Ry/Rz rotational stiffness coefficient, range: 0 ~ 20 (Nm/rad)

        # Attention: for M and J, smaller value means less effort to drive the arm, but may also be less stable, please be careful.
        M = float(0.03)  #  x/y/z equivalent mass; range: 0.02 ~ 1 kg
        J = (
            M * 0.01
        )  #  Rx/Ry/Rz equivalent moment of inertia, range: 1e-4 ~ 0.01 (Kg*m^2)

        c_axis = [1, 1, 1, 0, 0, 0]  # set z axis as compliant axis
        ref_frame = 0  # 0 : base , 1 : tool

        self._check_xarm_code(
            "set_ft_sensor_admittance_parameters",
            self.xarm_api.set_ft_sensor_admittance_parameters(
                [M, M, M, J, J, J], [K_pos, K_pos, K_pos, K_ori, K_ori, K_ori], [0] * 6
            ),
        )  # B(damping) is reserved, give zeros
        self._check_xarm_code(
            "set_ft_sensor_admittance_parameters",
            self.xarm_api.set_ft_sensor_admittance_parameters(ref_frame, c_axis),
        )
        # will overwrite previous sensor zero and payload configuration
        # self.xarm_api.set_ft_sensor_zero() # remove this if zero_offset and payload already identified & compensated!
        # Todo: in the super().__init__(), there is a line of self.xarm_api.set_ft_sensor_zero(). It might be better to remove it to avoid overwriting the admittance control configuration.
        time.sleep(0.2)  # wait for writing zero operation to take effect, do not remove

        self.xarm_api.clean_error()
        self.xarm_api.set_mode(6)
        self.xarm_api.set_state(0)
        self._check_xarm_code("set_ft_sensor_mode", self.xarm_api.set_ft_sensor_mode(1))
        self._check_xarm_code("set_state", self.xarm_api.set_state(0))
        print(f"[{self.__class__.__name__}] Enabled xArm7 admittance control.")

    def close(self):
        self.xarm_api.set_ft_sensor_mode(0)
        self.xarm_api.set_ft_sensor_enable(0)
        super().close()
