from pathlib import Path

import torch

from robo_manip_baselines.policy.wrench_predictor4.WrenchPredictor4Model import (
    WrenchPredictor4Model,
)

LEGACY_FIXED_ARGS = {
    "condition_image_feature": True,
    "condition_action": True,
    "pb_conditioning": "token",
    "pb_normalization_object_ids": [],
}


def load_policy(
    checkpoint_path: Path,
    model_meta_info: dict,
    device: torch.device,
) -> WrenchPredictor4Model:
    policy_args = dict(model_meta_info["policy"]["args"])
    for key, expected_value in LEGACY_FIXED_ARGS.items():
        actual_value = policy_args.pop(key, expected_value)
        assert actual_value == expected_value, (key, actual_value, expected_value)

    policy = WrenchPredictor4Model(**policy_args).to(device)
    policy.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )
    )
    policy.requires_grad_(False)
    policy.eval()
    return policy
