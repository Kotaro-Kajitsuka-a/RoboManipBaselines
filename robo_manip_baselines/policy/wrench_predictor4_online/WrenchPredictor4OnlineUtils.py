import pickle
from pathlib import Path

import numpy as np
import torch

from robo_manip_baselines.policy.wrench_predictor4.WrenchPredictor4Dataset import (
    WrenchPredictor4Dataset,
)
from robo_manip_baselines.policy.wrench_predictor4.WrenchPredictor4Model import (
    WrenchPredictor4Model,
)


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
    policy = WrenchPredictor4Model(**model_meta_info["policy"]["args"]).to(device)
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
