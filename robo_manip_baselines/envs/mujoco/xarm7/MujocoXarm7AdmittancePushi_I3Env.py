from .MujocoXarm7AdmittancePushiIEnvBase import MujocoXarm7AdmittancePushiIEnvBase


class MujocoXarm7AdmittancePushi_I3Env(MujocoXarm7AdmittancePushiIEnvBase):
    metadata = MujocoXarm7AdmittancePushiIEnvBase.metadata.copy()
    xml_filename = "env_xarm7_admittance_pushi_I3.xml"
    world_idx_range = range(300, 400)
