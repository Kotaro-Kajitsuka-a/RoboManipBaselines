import gymnasium as gym

from .OperationRealXarm7FixedGripperDemo import (
    OperationRealXarm7FixedGripperDemo,
)


class OperationRealXarm7PlanarConstraintDemo(OperationRealXarm7FixedGripperDemo):
    def setup_env(self, render_mode="human"):
        self.env = gym.make(
            "robo_manip_baselines/RealXarm7PlanarConstraintDemoEnv-v0",
            robot_ip=self.robot_ip,
            camera_ids=self.camera_ids,
            gelsight_ids=self.gelsight_ids,
        )
