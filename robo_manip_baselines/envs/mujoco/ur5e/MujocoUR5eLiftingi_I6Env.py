from .MujocoUR5eLiftingiEnvBase import MujocoUR5eLiftingiEnvBase


class MujocoUR5eLiftingi_I6Env(MujocoUR5eLiftingiEnvBase):
    metadata = MujocoUR5eLiftingiEnvBase.metadata.copy()
    xml_filename = "env_ur5e_liftingi_I6.xml"
    world_idx_range = range(600, 700)
