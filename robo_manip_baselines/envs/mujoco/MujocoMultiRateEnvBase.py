import mujoco
import numpy as np

from .MujocoEnvBase import MujocoEnvBase


class MujocoMultiRateEnvBase(MujocoEnvBase):
    sim_timestep = 0.001
    frame_skip = 32
    admittance_timestep = 0.004
    control_timestep = frame_skip * sim_timestep
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": int(np.round(1 / (MujocoEnvBase.sim_timestep * frame_skip))),
    }

    def __init__(self, xml_file, init_qpos, **kwargs):
        MujocoEnvBase.__init__(self, xml_file, init_qpos, **kwargs)
        self.model.opt.timestep = self.sim_timestep
        self.metadata["render_fps"] = int(np.round(1 / self.dt))
        self._n_physics_per_admittance = int(
            np.round(self.admittance_timestep / self.sim_timestep)
        )
        assert np.isclose(
            self._n_physics_per_admittance * self.sim_timestep,
            self.admittance_timestep,
        ), f"[{self.__class__.__name__}] admittance_timestep must be an integer multiple of sim_timestep."
        assert (
            self.frame_skip % self._n_physics_per_admittance == 0
        ), f"[{self.__class__.__name__}] frame_skip must be divisible by the admittance interval."

    def do_simulation(self, ctrl, n_frames) -> None:
        action = np.asarray(ctrl, dtype=np.float64)
        assert (
            n_frames == self.frame_skip
        ), f"[{self.__class__.__name__}] Multi-rate control expects n_frames == frame_skip."
        self._on_policy_step(action)

        for substep_idx in range(n_frames):
            mujoco.mj_step1(self.model, self.data)

            if substep_idx % self._n_physics_per_admittance == 0:
                self._on_admittance_step()

            self._apply_ctrl()
            mujoco.mj_step2(self.model, self.data)

        # As of MuJoCo 2.0, force-related quantities like cacc are not computed
        # unless there's a force sensor in the model.
        # See https://github.com/openai/gym/issues/1541
        mujoco.mj_rnePostConstraint(self.model, self.data)

    def _on_policy_step(self, action):
        raise NotImplementedError

    def _on_admittance_step(self):
        raise NotImplementedError

    def _apply_ctrl(self):
        raise NotImplementedError
