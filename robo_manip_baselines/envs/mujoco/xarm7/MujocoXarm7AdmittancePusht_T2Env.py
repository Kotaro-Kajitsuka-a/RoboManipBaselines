from .MujocoXarm7AdmittancePushtTEnvBase import MujocoXarm7AdmittancePushtTEnvBase


class MujocoXarm7AdmittancePusht_T2Env(MujocoXarm7AdmittancePushtTEnvBase):
    metadata = MujocoXarm7AdmittancePushtTEnvBase.metadata.copy()
    xml_filename = "env_xarm7_admittance_pusht_T2.xml"
    world_idx_range = range(200, 300)
