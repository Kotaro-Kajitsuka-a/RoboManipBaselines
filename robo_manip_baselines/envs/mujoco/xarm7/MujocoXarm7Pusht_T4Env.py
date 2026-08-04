from .MujocoXarm7PushtTEnvBase import MujocoXarm7PushtTEnvBase


class MujocoXarm7Pusht_T4Env(MujocoXarm7PushtTEnvBase):
    xml_filename = "env_xarm7_pusht_T4.xml"
    world_idx_range = range(400, 500)
