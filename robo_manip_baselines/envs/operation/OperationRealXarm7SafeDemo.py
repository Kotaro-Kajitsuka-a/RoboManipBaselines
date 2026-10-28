import gymnasium as gym

from .OperationRealXarm7Demo import OperationRealXarm7Demo


class OperationRealXarm7SafeDemo(OperationRealXarm7Demo):
    def setup_env(self, render_mode="human"):
        self.env = gym.make(
            "robo_manip_baselines/RealXarm7SafeDemoEnv-v0",
            robot_ip=self.robot_ip,
            camera_ids=self.camera_ids,
            gelsight_ids=self.gelsight_ids,
            has_gripper=self.has_gripper,
        )
