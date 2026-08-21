import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import roots_hermitenorm

from robo_manip_baselines.common import DataKey
from robo_manip_baselines.policy.wrench_predictor4.WrenchPredictor4Dataset import (
    WrenchPredictor4Dataset,
)
from robo_manip_baselines.policy.wrench_predictor4.WrenchPredictor4Model import (
    WrenchPredictor4Model,
)

ONLINE_PB_STD_KEY = DataKey.ONLINE_PB_STD
GAUSSIAN_POINTS_PER_PB_DIM = 16


def resolve_gaussian_num_points(pb_dim: int, num_points: int | None) -> int:
    assert pb_dim >= 1, pb_dim
    resolved = GAUSSIAN_POINTS_PER_PB_DIM * pb_dim if num_points is None else num_points
    assert resolved >= 3, resolved
    return resolved


class GaussianBeliefOnlinePb:
    """One-dimensional Gaussian PB belief updated by Gauss-Hermite moments."""

    def __init__(
        self,
        initial_mean: np.ndarray,
        initial_std: float,
        num_points: int,
        beta: float,
        device: torch.device,
    ):
        assert initial_mean.shape == (1,), initial_mean.shape
        assert initial_std > 0.0, initial_std
        assert num_points >= 3, num_points
        assert beta > 0.0, beta

        nodes, prior_weights = roots_hermitenorm(num_points)
        prior_weights = prior_weights / prior_weights.sum()
        self.nodes = torch.tensor(nodes, dtype=torch.float32, device=device)
        self.log_prior_weights = torch.tensor(
            np.log(prior_weights),
            dtype=torch.float32,
            device=device,
        )
        self.mean = torch.tensor(initial_mean, dtype=torch.float32, device=device)
        self.std = torch.full_like(self.mean, initial_std)
        self.beta = beta

    @property
    def num_points(self) -> int:
        return self.nodes.shape[0]

    def get_candidates(self) -> torch.Tensor:
        return self.mean.unsqueeze(0) + self.std.unsqueeze(0) * self.nodes.unsqueeze(1)

    @torch.no_grad()
    def update(self, losses: torch.Tensor) -> torch.Tensor:
        assert losses.shape == (self.num_points,), losses.shape
        assert torch.isfinite(losses).all(), losses

        candidates = self.get_candidates()
        weights = torch.softmax(self.log_prior_weights - self.beta * losses, dim=0)
        new_mean = torch.sum(weights.unsqueeze(1) * candidates, dim=0)
        new_variance = torch.sum(
            weights.unsqueeze(1) * (candidates - new_mean).square(),
            dim=0,
        )
        self.mean.copy_(new_mean)
        self.std.copy_(torch.sqrt(new_variance.clamp_min(0.0)))
        return weights


def calculate_pb_candidate_losses(
    policy: WrenchPredictor4Model,
    batch: dict[str, torch.Tensor],
    pb_candidates: torch.Tensor,
    wrench_loss_weight: float,
) -> torch.Tensor:
    """Evaluate every PB candidate against one observed WP4 window."""
    num_candidates = pb_candidates.shape[0]
    assert batch["state"].shape[0] == 1, batch["state"].shape
    candidate_batch = {
        key: value.expand(num_candidates, *value.shape[1:])
        for key, value in batch.items()
    }
    prediction = policy(candidate_batch, pb_candidates)
    start = policy.n_obs_steps
    pose_loss = (
        F.mse_loss(
            prediction["image_feature"][:, start:],
            candidate_batch["image_feature"][:, start:],
            reduction="none",
        )
        .flatten(start_dim=1)
        .mean(dim=1)
    )
    wrench_loss = (
        F.mse_loss(
            prediction["wrench"][:, start:],
            candidate_batch["wrench"][:, start:],
            reduction="none",
        )
        .flatten(start_dim=1)
        .mean(dim=1)
    )
    return pose_loss + wrench_loss_weight * wrench_loss


def load_model_meta_info(checkpoint_path: Path) -> dict:
    checkpoint_path = checkpoint_path.resolve()
    assert checkpoint_path.is_file(), checkpoint_path

    meta_info_path = checkpoint_path.parent / "model_meta_info.pkl"
    assert meta_info_path.is_file(), meta_info_path
    with meta_info_path.open("rb") as file:
        model_meta_info = pickle.load(file)

    assert model_meta_info["policy"]["name"] == "WrenchPredictor4"
    policy_args = model_meta_info["policy"]["args"]
    pb_dim = policy_args["pb_dim"]
    assert model_meta_info["material_property"]["pb_dim"] == pb_dim, (
        model_meta_info["material_property"]["pb_dim"],
        pb_dim,
    )
    object_key_to_id = model_meta_info["material_property"]["object_key_to_id"]
    assert policy_args["num_objects"] == len(object_key_to_id), (
        policy_args["num_objects"],
        len(object_key_to_id),
    )
    return model_meta_info


def load_policy(
    checkpoint_path: Path,
    model_meta_info: dict,
    device: torch.device,
) -> WrenchPredictor4Model:
    policy_args = model_meta_info["policy"]["args"].copy()
    # The state-based EefPose checkpoint records newer configuration fields whose
    # default values are exactly the legacy token-conditioning architecture.
    assert policy_args.pop("condition_image_feature", True) is True
    assert policy_args.pop("condition_action", True) is True
    assert policy_args.pop("pb_conditioning", "token") == "token"
    assert policy_args.pop("pb_normalization_object_ids", []) == []
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


def load_pb_table(
    checkpoint_path: Path,
    model_meta_info: dict | None = None,
) -> tuple[np.ndarray, dict[str, int]]:
    checkpoint_path = checkpoint_path.resolve()
    if model_meta_info is None:
        model_meta_info = load_model_meta_info(checkpoint_path)

    object_key_to_id = model_meta_info["material_property"]["object_key_to_id"]
    assert object_key_to_id == WrenchPredictor4Dataset.OBJECT_KEY_TO_ID, (
        object_key_to_id,
        WrenchPredictor4Dataset.OBJECT_KEY_TO_ID,
    )

    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    material_property = state_dict["material_property.weight"]
    pb_dim = model_meta_info["material_property"]["pb_dim"]
    assert material_property.shape == (len(object_key_to_id), pb_dim), (
        material_property.shape,
        len(object_key_to_id),
        pb_dim,
    )

    pb_table = material_property.detach().numpy().astype(np.float32)
    return pb_table, object_key_to_id


def load_pb(
    checkpoint_path: Path,
    object_id: int,
    model_meta_info: dict | None = None,
) -> tuple[np.ndarray, str]:
    pb_table, object_key_to_id = load_pb_table(checkpoint_path, model_meta_info)
    object_id_to_key = {
        mapped_id: object_key for object_key, mapped_id in object_key_to_id.items()
    }
    assert object_id in object_id_to_key, (object_id, sorted(object_id_to_key))
    return pb_table[object_id], object_id_to_key[object_id]
