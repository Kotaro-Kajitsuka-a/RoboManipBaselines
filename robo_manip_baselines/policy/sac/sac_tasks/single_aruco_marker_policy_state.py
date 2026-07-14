import numpy as np
import torch

from robo_manip_baselines.common import DataKey

from ..gripper_utils import (
    convert_gripper_positions_to_maniskill,
    convert_gripper_velocities_to_maniskill,
)


def get_policy_state(task, rollout):
    qpos = np.asarray(
        rollout.motion_manager.get_data(DataKey.MEASURED_JOINT_POS, rollout.obs),
        dtype=np.float32,
    ).reshape(-1)
    qvel = np.asarray(
        rollout.motion_manager.get_data(DataKey.MEASURED_JOINT_VEL, rollout.obs),
        dtype=np.float32,
    ).reshape(-1)
    if qpos.size != rollout.action_dim:
        raise ValueError(
            f"[{rollout.__class__.__name__}] measured joint pos must have dim "
            f"{rollout.action_dim}, got {qpos.size}."
        )
    if qvel.size != rollout.action_dim:
        raise ValueError(
            f"[{rollout.__class__.__name__}] measured joint vel must have dim "
            f"{rollout.action_dim}, got {qvel.size}."
        )

    qpos_ms = convert_gripper_positions_to_maniskill(
        qpos.copy(), rollout._gripper_joint_indices
    )
    qvel_ms = convert_gripper_velocities_to_maniskill(
        qvel.copy(), rollout._gripper_joint_indices
    )
    rollout._latest_joint_pos_tensor = torch.as_tensor(
        qpos_ms, dtype=torch.float32, device=rollout.device
    )

    left_idx = rollout._get_arm_joint_indices(0)
    right_idx = rollout._get_arm_joint_indices(1)
    extra = task.get_extra_state()
    if "trash_bin_position" in rollout.state_keys:
        position_key = "trash_bin_position"
        rotation_key = "trash_bin_rotation_6d"
        prev_position_key = "prev_trash_bin_position"
        prev_rotation_key = "prev_trash_bin_rotation_6d"
    else:
        position_key = "marker_position"
        rotation_key = "marker_rotation_6d"
        prev_position_key = "prev_marker_position"
        prev_rotation_key = "prev_marker_rotation_6d"

    parts = [
        qpos_ms[left_idx],
        qvel_ms[left_idx],
        qpos_ms[right_idx],
        qvel_ms[right_idx],
        extra[position_key],
        extra[rotation_key],
    ]
    if prev_position_key in rollout.state_keys:
        parts += [
            extra[prev_position_key],
            extra[prev_rotation_key],
        ]
    state_vector = np.concatenate(
        [np.asarray(part, dtype=np.float32).reshape(-1) for part in parts]
    ).astype(np.float32)
    if state_vector.size != rollout.state_dim:
        raise ValueError(
            f"[{rollout.__class__.__name__}] State dimension mismatch. "
            f"Constructed {state_vector.size}, expected {rollout.state_dim}."
        )
    return torch.as_tensor(
        state_vector, dtype=torch.float32, device=rollout.device
    ).unsqueeze(0)
