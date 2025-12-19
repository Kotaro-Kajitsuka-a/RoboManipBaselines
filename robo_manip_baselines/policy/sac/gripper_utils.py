from typing import Iterable, Optional

import numpy as np
import torch


def gripper_q_robomanip_to_maniskill(q_robomanip: float) -> float:
    """Convert RoboManip gripper position scalar to ManiSkill scale."""

    return (q_robomanip - 850.0) / (-1000.0)


def gripper_qvel_robomanip_to_maniskill(qvel_robomanip: float) -> float:
    """Convert RoboManip gripper velocity scalar to ManiSkill scale."""

    return qvel_robomanip / (-1000.0)


def gripper_q_maniskill_to_robomanip(q_maniskill: float) -> float:
    """Convert ManiSkill gripper position scalar to RoboManip scale."""

    return q_maniskill * (-1000.0) + 850.0


def _normalize_indices(
    gripper_joint_indices: Optional[Iterable[int]],
) -> np.ndarray:
    if gripper_joint_indices is None:
        return np.array([], dtype=np.int64)
    indices = np.asarray(list(gripper_joint_indices), dtype=np.int64).reshape(-1)
    return indices


def convert_gripper_positions_to_maniskill(
    values: np.ndarray, gripper_joint_indices: Optional[Iterable[int]]
) -> np.ndarray:
    indices = _normalize_indices(gripper_joint_indices)
    if indices.size == 0:
        return values
    converted = values.copy()
    for idx in indices:
        converted[idx] = gripper_q_robomanip_to_maniskill(float(converted[idx]))
    return converted


def convert_gripper_velocities_to_maniskill(
    values: np.ndarray, gripper_joint_indices: Optional[Iterable[int]]
) -> np.ndarray:
    indices = _normalize_indices(gripper_joint_indices)
    if indices.size == 0:
        return values
    converted = values.copy()
    for idx in indices:
        converted[idx] = gripper_qvel_robomanip_to_maniskill(float(converted[idx]))
    return converted


def convert_gripper_tensor_to_robomanip(
    tensor: torch.Tensor, gripper_joint_indices: Optional[Iterable[int]]
) -> torch.Tensor:
    indices = _normalize_indices(gripper_joint_indices)
    if indices.size == 0:
        return tensor
    for idx in indices:
        tensor[idx] = tensor[idx].new_tensor(
            gripper_q_maniskill_to_robomanip(float(tensor[idx].item()))
        )
    return tensor

if __name__=="__main__":
    print(gripper_q_robomanip_to_maniskill(119.0))
