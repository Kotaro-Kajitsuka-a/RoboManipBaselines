from types import SimpleNamespace

import numpy as np
import torch
from gymnasium.spaces import Box

from robo_manip_baselines.common import denormalize_data

from .SacPolicy import Actor
from .gripper_utils import (
    convert_gripper_tensor_to_robomanip,
)


def build_policy_env(model_meta_info):
    obs_dim = len(model_meta_info["state"]["example"])
    action_dim = len(model_meta_info["action"]["example"])
    action_low = model_meta_info["action"].get("low")
    action_high = model_meta_info["action"].get("high")
    if action_low is None or action_high is None:
        action_low = np.full(action_dim, -1.0, dtype=np.float32)
        action_high = np.full(action_dim, 1.0, dtype=np.float32)
    else:
        action_low = np.asarray(action_low, dtype=np.float32).reshape(-1)
        action_high = np.asarray(action_high, dtype=np.float32).reshape(-1)
    if action_low.size != action_dim or action_high.size != action_dim:
        raise ValueError(
            f"SAC action bounds mismatch: low={action_low.size}, "
            f"high={action_high.size}, expected={action_dim}."
        )
    return SimpleNamespace(
        single_observation_space=Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        ),
        single_action_space=Box(low=action_low, high=action_high, dtype=np.float32),
    )


def build_policy(model_meta_info) -> Actor:
    return Actor(build_policy_env(model_meta_info))


def infer_policy(rollout):
    if rollout.policy_action_buf is None or len(rollout.policy_action_buf) == 0:
        obs_tensor = rollout.get_state()
        rollout._latest_policy_state_tensor = obs_tensor

        with torch.no_grad():
            if rollout._deterministic:
                raw_action = rollout.policy.get_eval_action(obs_tensor)
            else:
                raw_action, _, _ = rollout.policy.get_action(obs_tensor)

        raw_action = raw_action.squeeze(0)
        clipped_action = torch.clamp(
            raw_action, rollout._normalized_action_low, rollout._normalized_action_high
        )

        if clipped_action.numel() != rollout.action_dim:
            raise ValueError(
                f"[{rollout.__class__.__name__}] Unexpected action length "
                f"{clipped_action.numel()} for action_dim {rollout.action_dim}."
            )

        normalized_span = rollout._normalized_action_high - rollout._normalized_action_low
        delta_scale = (clipped_action - rollout._normalized_action_low) / normalized_span
        denormalized_delta = rollout._action_delta_low + delta_scale * (
            rollout._action_delta_high - rollout._action_delta_low
        )

        direct_joint_command = rollout._latest_joint_pos_tensor + denormalized_delta
        direct_joint_command = torch.max(
            torch.min(direct_joint_command, rollout._joint_position_high),
            rollout._joint_position_low,
        )
        direct_joint_command = direct_joint_command.clone()
        direct_joint_command = convert_gripper_tensor_to_robomanip(
            direct_joint_command, rollout._gripper_joint_indices
        )

        physical_np = direct_joint_command.detach().cpu().numpy().astype(np.float64)
        physical_np[rollout._gripper_joint_indices] = rollout._fixed_gripper_command
        rollout.policy_action_buf = [physical_np]

    rollout.policy_action = denormalize_data(
        rollout.policy_action_buf.pop(0), rollout.model_meta_info["action"]
    )
    rollout.append_state_action_csv(
        rollout._latest_policy_state_tensor, rollout.policy_action
    )
    rollout.policy_action_list = np.concatenate(
        [rollout.policy_action_list, rollout.policy_action[np.newaxis]]
    )
