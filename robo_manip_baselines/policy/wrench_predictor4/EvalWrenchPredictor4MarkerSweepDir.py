import os

import torch

from robo_manip_baselines.policy.diffusion_world_model.EvalDiffusionWorldModelMarkerSweepDir import (
    EvalDiffusionWorldModelMarkerSweepDir,
)
from robo_manip_baselines.policy.diffusion_world_model.EvalDiffusionWorldModelSweepCommon import (
    parse_sweep_argument,
)
from robo_manip_baselines.policy.wrench_predictor4.WrenchPredictor4Model import (
    WrenchPredictor4Model,
)


class EvalWrenchPredictor4MarkerSweepDir(EvalDiffusionWorldModelMarkerSweepDir):
    def get_output_name(self, rmb_dir_name):
        return f"{rmb_dir_name}_wrench_predictor4_marker_sweep"

    def setup_policy(self, checkpoint):
        self.policy = WrenchPredictor4Model(**self.model_meta_info["policy"]["args"])
        self.policy.load_state_dict(
            torch.load(checkpoint, map_location=self.device, weights_only=True)
        )
        self.policy.to(self.device)
        self.policy.eval()

    def get_heatmap_png_name(self, checkpoint_stem):
        return f"{checkpoint_stem}_wrench_predictor4_marker_sweep_heatmap.png"


if __name__ == "__main__":
    evaluator = EvalWrenchPredictor4MarkerSweepDir(**vars(parse_sweep_argument()))
    evaluator.run()
