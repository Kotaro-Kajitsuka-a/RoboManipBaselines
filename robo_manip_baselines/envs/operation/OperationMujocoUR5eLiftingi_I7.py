import gymnasium as gym

from robo_manip_baselines.common import GraspPhaseBase


class GraspPhase(GraspPhaseBase):
    def set_target(self):
        self.set_target_open()


class OperationMujocoUR5eLiftingi_I7:
    def set_additional_args(self, parser):
        super().set_additional_args(parser)
        parser.add_argument(
            "--allow_out_of_range_world_idx",
            action="store_true",
            help="whether to allow world indexes outside the range assigned to this environment",
        )

    def setup_env(self, render_mode="human"):
        self.env = gym.make(
            "robo_manip_baselines/MujocoUR5eLiftingi_I7Env-v0",
            render_mode=render_mode,
            allow_out_of_range_world_idx=self.args.allow_out_of_range_world_idx,
        )

    def get_pre_motion_phases(self):
        return [
            GraspPhase(self),
        ]
