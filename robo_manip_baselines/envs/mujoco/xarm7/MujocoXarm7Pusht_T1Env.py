from .MujocoXarm7PushtTEnvBase import MujocoXarm7PushtTEnvBase


class MujocoXarm7Pusht_T1Env(MujocoXarm7PushtTEnvBase):
    xml_filename = "env_xarm7_pusht_T1.xml"
    world_idx_range = range(100, 200)
