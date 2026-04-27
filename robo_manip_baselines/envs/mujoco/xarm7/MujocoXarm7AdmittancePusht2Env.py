from os import path

import numpy as np

from .MujocoXarm7AdmittanceEnvBase import MujocoXarm7AdmittanceEnvBase
from .MujocoXarm7PushtEnv import MujocoXarm7PushtEnv


class MujocoXarm7AdmittancePusht2Env(
    MujocoXarm7AdmittanceEnvBase, MujocoXarm7PushtEnv
):
    def __init__(self, **kwargs):
        MujocoXarm7AdmittanceEnvBase.__init__(
            self,
            path.join(
                path.dirname(__file__),
                "../../assets/mujoco/envs/xarm7/env_xarm7_pusht2.xml",
            ),
            np.array([0.0, 0.0, 0.0, 0.8, 0.0, 0.8, 0.0, *[0.0] * 6]),
            **kwargs,
        )
        self.original_tblock_pos = self.model.body("tblock").pos.copy()
        self.tblock_pos_offsets = np.array(
            [
                [0.0, -0.06, 0.0],
                [0.0, -0.03, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.03, 0.0],
                [0.0, 0.06, 0.0],
                [0.0, 0.09, 0.0],
            ]
        )
