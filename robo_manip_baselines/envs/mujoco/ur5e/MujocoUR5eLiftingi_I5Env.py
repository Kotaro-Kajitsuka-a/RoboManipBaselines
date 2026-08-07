from .MujocoUR5eLiftingiEnvBase import MujocoUR5eLiftingiEnvBase


class MujocoUR5eLiftingi_I5Env(MujocoUR5eLiftingiEnvBase):
    metadata = MujocoUR5eLiftingiEnvBase.metadata.copy()
    xml_filename = "env_ur5e_liftingi_I5.xml"
    world_idx_range = range(500, 600)
