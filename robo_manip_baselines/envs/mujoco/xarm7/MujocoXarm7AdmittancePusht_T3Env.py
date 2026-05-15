from .MujocoXarm7AdmittancePushtTEnvBase import MujocoXarm7AdmittancePushtTEnvBase


class MujocoXarm7AdmittancePusht_T3Env(MujocoXarm7AdmittancePushtTEnvBase):
    metadata = MujocoXarm7AdmittancePushtTEnvBase.metadata.copy()
    xml_filename = "env_xarm7_admittance_pusht_T3.xml"
    world_idx_range = range(300, 400)
