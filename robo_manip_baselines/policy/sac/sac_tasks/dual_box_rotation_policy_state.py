import numpy as np
import torch

from robo_manip_baselines.common import DataKey

from ..gripper_utils import (
    convert_gripper_positions_to_maniskill,
    convert_gripper_velocities_to_maniskill,
)
from ..state_utils import get_adjusted_measured_eef_pose


def get_box_pose(task, rollout) -> np.ndarray:
    extra_state = task.get_extra_state()
    assert isinstance(extra_state, dict)
    box_pose = np.asarray(extra_state["box_pose"], dtype=np.float32).reshape(-1)
    expected_dim = rollout.extra_state_dims.get("box_pose")
    if expected_dim is not None and box_pose.size != expected_dim:
        raise ValueError(
            f"[{rollout.__class__.__name__}] box_pose dimension "
            f"{box_pose.size} != {expected_dim}."
        )
    return box_pose


def get_box_pushpoint(task, rollout, box_pose: np.ndarray) -> np.ndarray:
    pushpoint = task._get_box_pushpoint(box_pose)
    pushpoint = np.asarray(pushpoint, dtype=np.float32).reshape(-1)
    expected_dim = rollout.extra_state_dims.get("box_pushpoint")
    if expected_dim is not None and pushpoint.size != expected_dim:
        raise ValueError(
            f"[{rollout.__class__.__name__}] box_pushpoint dimension "
            f"{pushpoint.size} != {expected_dim}."
        )
    return pushpoint


def get_policy_state(task, rollout):
    qpos = rollout.motion_manager.get_data(DataKey.MEASURED_JOINT_POS, rollout.obs)
    qvel = rollout.motion_manager.get_data(DataKey.MEASURED_JOINT_VEL, rollout.obs)

    qpos_ms = convert_gripper_positions_to_maniskill(
        qpos.astype(np.float32).copy(), rollout._gripper_joint_indices
    )
    qvel_ms = convert_gripper_velocities_to_maniskill(
        qvel.astype(np.float32).copy(), rollout._gripper_joint_indices
    )
    rollout._latest_joint_pos_tensor = torch.as_tensor(
        qpos_ms, dtype=torch.float32, device=rollout.device
    )

    left_idx = rollout._get_arm_joint_indices(0)
    right_idx = rollout._get_arm_joint_indices(1)
    box_pose = get_box_pose(task, rollout)

    measured_eef_pose = rollout.motion_manager.get_data(
        DataKey.MEASURED_EEF_POSE, rollout.obs
    )
    left_tcp_pose_6d, right_tcp_pose_6d = get_adjusted_measured_eef_pose(
        measured_eef_pose
    )

    parts = [
        qpos_ms[left_idx],
        qvel_ms[left_idx],
        qpos_ms[right_idx],
        qvel_ms[right_idx],
        box_pose,
        left_tcp_pose_6d,
        right_tcp_pose_6d,
        get_box_pushpoint(task, rollout, box_pose),
    ]
    state_vector = np.concatenate(
        [np.asarray(part, dtype=np.float32).reshape(-1) for part in parts]
    ).astype(np.float32)
    expected_state_dim = len(rollout.model_meta_info["state"]["example"])
    if state_vector.size != expected_state_dim:
        raise ValueError(
            f"[{rollout.__class__.__name__}] State dimension mismatch. "
            f"Constructed {state_vector.size} elements, "
            f"but model_meta_info expects {expected_state_dim}."
        )

    return torch.tensor(
        state_vector, dtype=torch.float32, device=rollout.device
    ).unsqueeze(0)
