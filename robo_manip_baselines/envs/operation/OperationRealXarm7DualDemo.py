import gymnasium as gym

from robo_manip_baselines.common import GraspPhaseBase


class GraspPhase(GraspPhaseBase):
    def set_target(self):
        # Skip sending any gripper command before rollout.
        self.duration = 0.0

    def pre_update(self):
        # Do nothing so the hardware grippers stay at their current positions.
        pass

    def check_transition(self):
        # Move to the next phase immediately.
        return True

        
class OperationRealXarm7DualDemo:
    def __init__(self, robot_ip_left, robot_ip_right, camera_ids, gelsight_ids=None):
        self.robot_ip_left = robot_ip_left
        self.robot_ip_right = robot_ip_right
        self.camera_ids = camera_ids
        self.gelsight_ids = gelsight_ids
        super().__init__()

    def setup_env(self, render_mode="human"):
        self.env = gym.make(
            "robo_manip_baselines/RealXarm7DualDemoEnv-v0",
            robot_ip_left=self.robot_ip_left,
            robot_ip_right=self.robot_ip_right,
            camera_ids=self.camera_ids,
            gelsight_ids=self.gelsight_ids,
        )

    def get_pre_motion_phases(self):
        return [GraspPhase(self)]
