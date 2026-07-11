import numpy as np
import torch

from robo_manip_baselines.common import DataKey

from ..gripper_utils import (
    convert_gripper_positions_to_maniskill,
    convert_gripper_velocities_to_maniskill,
)
from ..state_utils import get_adjusted_measured_eef_pose

_LEFT_TCP_OFFSET_M = np.array([0.0, -0.3291, 0.27589], dtype=np.float32)
_RIGHT_TCP_OFFSET_M = np.array([0.0, 0.3291, 0.27589], dtype=np.float32)
_TCP_ROTATION_ADJUST = np.array(
    [
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float32,
)


def _rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    rot6d = np.asarray(rot6d, dtype=np.float32).reshape(6)
    x_axis = rot6d[0:3]
    y_axis = rot6d[3:6]
    x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-8)
    y_axis = y_axis - np.dot(x_axis, y_axis) * x_axis
    y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-8)
    z_axis = np.cross(x_axis, y_axis)
    return np.stack([x_axis, y_axis, z_axis], axis=1)


def _matrix_to_rot6d(rotation: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=np.float32).reshape(3, 3)
    return np.concatenate([rotation[:, 0], rotation[:, 1]]).astype(np.float32)


def _adjust_tcp_pose(tcp_pose_6d: np.ndarray, offset: np.ndarray) -> np.ndarray:
    tcp_pose_6d = np.asarray(tcp_pose_6d, dtype=np.float32).reshape(9)
    position = tcp_pose_6d[0:3]
    rotation = _rot6d_to_matrix(tcp_pose_6d[3:9])
    adjusted_position = position + rotation @ offset
    adjusted_rotation = rotation @ _TCP_ROTATION_ADJUST
    return np.concatenate(
        [adjusted_position, _matrix_to_rot6d(adjusted_rotation)]
    ).astype(np.float32)


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
    left_tcp_pose_6d = _adjust_tcp_pose(left_tcp_pose_6d, _LEFT_TCP_OFFSET_M)
    right_tcp_pose_6d = _adjust_tcp_pose(right_tcp_pose_6d, _RIGHT_TCP_OFFSET_M)

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
