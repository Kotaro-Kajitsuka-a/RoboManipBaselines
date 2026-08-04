from .MujocoXarm7PushtTEnvBase import MujocoXarm7PushtTEnvBase


class MujocoXarm7Pusht_T0Env(MujocoXarm7PushtTEnvBase):
    xml_filename = "env_xarm7_pusht_T0.xml"
    world_idx_range = range(0, 100)
