import numpy as np
import torch

from robo_manip_baselines.common import denormalize_data

from .PpoPolicy import ManiSkillPpoAgent
from .gripper_utils import convert_gripper_tensor_to_robomanip


def build_policy(state_dict, state_dim: int, action_dim: int) -> ManiSkillPpoAgent:
    obs_dim = int(state_dict["actor_mean.0.weight"].shape[1])
    ckpt_action_dim = int(state_dict["actor_logstd"].shape[-1])
    if obs_dim != state_dim or ckpt_action_dim != action_dim:
        raise ValueError(
            f"PPO checkpoint dims must be obs={state_dim}, action={action_dim}; "
            f"got obs={obs_dim}, action={ckpt_action_dim}."
        )
    return ManiSkillPpoAgent(obs_dim, action_dim)


def infer_policy(rollout):
    if rollout.policy_action_buf is None or len(rollout.policy_action_buf) == 0:
        obs_tensor = rollout.get_state()
        with torch.no_grad():
            raw_action = rollout.policy.get_action(
                obs_tensor, deterministic=rollout._deterministic
            )

        raw_action = raw_action.squeeze(0)
        clipped_action = torch.clamp(
            raw_action, rollout._normalized_action_low, rollout._normalized_action_high
        )
        delta_scale = (clipped_action - rollout._normalized_action_low) / (
            rollout._normalized_action_high - rollout._normalized_action_low
        )
        denormalized_delta = rollout._action_delta_low + delta_scale * (
            rollout._action_delta_high - rollout._action_delta_low
        )
        direct_joint_command = rollout._latest_joint_pos_tensor + denormalized_delta
        direct_joint_command = torch.max(
            torch.min(direct_joint_command, rollout._joint_position_high),
            rollout._joint_position_low,
        ).clone()
        direct_joint_command = convert_gripper_tensor_to_robomanip(
            direct_joint_command, rollout._gripper_joint_indices
        )

        physical_np = direct_joint_command.detach().cpu().numpy().astype(np.float64)
        physical_np[rollout._gripper_joint_indices] = rollout._fixed_gripper_command
        rollout.policy_action_buf = [physical_np]

    rollout.policy_action = denormalize_data(
        rollout.policy_action_buf.pop(0), rollout.model_meta_info["action"]
    )
    rollout.policy_action[rollout._gripper_joint_indices] = (
        rollout._fixed_gripper_command
    )
    rollout.policy_action_list = np.concatenate(
        [rollout.policy_action_list, rollout.policy_action[np.newaxis]]
    )
