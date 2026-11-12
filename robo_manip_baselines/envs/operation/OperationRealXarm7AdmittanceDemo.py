import gymnasium as gym
import numpy as np

from robo_manip_baselines.common import GraspPhaseBase

from .OperationRealXarm7Demo import OperationRealXarm7Demo


class AdmittanceOpenGripperPhase(GraspPhaseBase):
    def start(self):
        super().start()
        print(f"[{self.op.__class__.__name__}] Press the 'n' key to close the gripper.")

    def set_target(self):
        self.gripper_joint_pos = np.array([800.0])

    def check_transition(self):
        return self.op.key == ord("n")


class AdmittanceGraspPhase(GraspPhaseBase):
    def set_target(self):
        self.gripper_joint_pos = np.array([0.0])
        self.duration = 0.5  # [s]


class OperationRealXarm7AdmittanceDemo(OperationRealXarm7Demo):
    def setup_env(self, render_mode="human"):
        self.env = gym.make(
            "robo_manip_baselines/RealXarm7AdmittanceDemoEnv-v0",
            robot_ip=self.robot_ip,
            camera_ids=self.camera_ids,
            gelsight_ids=self.gelsight_ids,
            has_gripper=self.has_gripper,
            pressure_sensor_ids=self.pressure_sensor_ids,
        )

    def get_pre_motion_phases(self):
        # return [AdmittanceOpenGripperPhase(self), AdmittanceGraspPhase(self)]
        return [AdmittanceGraspPhase(self)]
