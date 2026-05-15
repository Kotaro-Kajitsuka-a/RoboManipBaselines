from .MujocoXarm7AdmittancePushtTEnvBase import MujocoXarm7AdmittancePushtTEnvBase


class MujocoXarm7AdmittancePusht_T4Env(MujocoXarm7AdmittancePushtTEnvBase):
    metadata = MujocoXarm7AdmittancePushtTEnvBase.metadata.copy()
    xml_filename = "env_xarm7_admittance_pusht_T4.xml"
    world_idx_range = range(400, 500)
