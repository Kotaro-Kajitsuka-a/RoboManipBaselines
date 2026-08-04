from .MujocoXarm7PushtTEnvBase import MujocoXarm7PushtTEnvBase


class MujocoXarm7Pusht_T2Env(MujocoXarm7PushtTEnvBase):
    xml_filename = "env_xarm7_pusht_T2.xml"
    world_idx_range = range(200, 300)
