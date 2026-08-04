from .MujocoXarm7PushtTEnvBase import MujocoXarm7PushtTEnvBase


class MujocoXarm7Pusht_T3Env(MujocoXarm7PushtTEnvBase):
    xml_filename = "env_xarm7_pusht_T3.xml"
    world_idx_range = range(300, 400)
