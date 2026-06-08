from .MujocoXarm7AdmittancePushiIEnvBase import MujocoXarm7AdmittancePushiIEnvBase


class MujocoXarm7AdmittancePushi_I0Env(MujocoXarm7AdmittancePushiIEnvBase):
    metadata = MujocoXarm7AdmittancePushiIEnvBase.metadata.copy()
    xml_filename = "env_xarm7_admittance_pushi_I0.xml"
    world_idx_range = range(0, 100)
