import csv
import json
import os
from pathlib import Path
from typing import Optional

import cv2
import matplotlib.pylab as plt
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from robo_manip_baselines.common import DataKey, RolloutBase, denormalize_data

from .gripper_utils import (
    gripper_q_maniskill_to_robomanip,
    gripper_q_robomanip_to_maniskill,
    gripper_qvel_robomanip_to_maniskill,
)


STATE_DIM = 32
ACTION_DIM = 16
LEFT_JOINT_IDX = np.arange(0, 8, dtype=np.int64)
RIGHT_JOINT_IDX = np.arange(8, 16, dtype=np.int64)
GRIPPER_JOINT_IDX = np.array([7, 15], dtype=np.int64)
ARM_JOINT_DELTA_LIMIT = 0.03
GRIPPER_JOINT_DELTA_LIMIT = 0.1
FIXED_GRIPPER_COMMAND = 119.0
PPO_DETERMINISTIC = True


def _layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class ManiSkillPpoAgent(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.critic = nn.Sequential(
            _layer_init(nn.Linear(obs_dim, 256)),
            nn.Tanh(),
            _layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            _layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            _layer_init(nn.Linear(256, 1)),
        )
        self.actor_mean = nn.Sequential(
            _layer_init(nn.Linear(obs_dim, 256)),
            nn.Tanh(),
            _layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            _layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            _layer_init(nn.Linear(256, action_dim), std=0.01 * np.sqrt(2)),
        )
        self.actor_logstd = nn.Parameter(torch.ones(1, action_dim) * -0.5)

    def get_action(self, obs, deterministic=False):
        action_mean = self.actor_mean(obs)
        if deterministic:
            return action_mean

        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        return probs.sample()


class RolloutRLPolicyDual(RolloutBase):
    """RealXarm7Dual-only PPO rollout with fixed 32D joint observations."""

    def set_additional_args(self, parser):
        parser.add_argument(
            "--rl-log-tsv",
            action="store_true",
            help="Log observations and actions next to the checkpoint.",
        )

    def setup_model_meta_info(self):
        super().setup_model_meta_info()
        self._validate_meta_info()

    def _validate_meta_info(self):
        if list(self.action_keys) != [DataKey.COMMAND_JOINT_POS]:
            raise ValueError(
                f"[{self.__class__.__name__}] action keys must be {[DataKey.COMMAND_JOINT_POS]}, "
                f"got {self.action_keys}"
            )
        if self.state_dim != STATE_DIM:
            raise ValueError(
                f"[{self.__class__.__name__}] state dim must be {STATE_DIM}, got {self.state_dim}."
            )
        if self.action_dim != ACTION_DIM:
            raise ValueError(
                f"[{self.__class__.__name__}] action dim must be {ACTION_DIM}, got {self.action_dim}."
            )

    def setup_policy(self):
        self.print_policy_info()
        state_dict = torch.load(self.args.checkpoint, map_location="cpu")
        if isinstance(state_dict, dict) and "actor" in state_dict:
            state_dict = state_dict["actor"]

        if "actor_mean.0.weight" not in state_dict or "actor_logstd" not in state_dict:
            raise KeyError(
                f"[{self.__class__.__name__}] PPO checkpoint does not contain actor_mean.0.weight and actor_logstd."
            )

        obs_dim = int(state_dict["actor_mean.0.weight"].shape[1])
        action_dim = int(state_dict["actor_logstd"].shape[-1])
        if obs_dim != STATE_DIM or action_dim != ACTION_DIM:
            raise ValueError(
                f"[{self.__class__.__name__}] checkpoint dims must be obs={STATE_DIM}, "
                f"action={ACTION_DIM}; got obs={obs_dim}, action={action_dim}."
            )

        self.policy = ManiSkillPpoAgent(obs_dim, action_dim)
        self.policy.load_state_dict(state_dict)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy.to(self.device)
        self.policy.eval()

        self._normalized_action_low = torch.full((ACTION_DIM,), -1.0, device=self.device)
        self._normalized_action_high = torch.full((ACTION_DIM,), 1.0, device=self.device)
        self._joint_position_low, self._joint_position_high = self._get_joint_limits()
        self._action_delta_low, self._action_delta_high = self._get_action_delta_limits()

        checkpoint_dir = Path(os.path.dirname(os.path.abspath(self.args.checkpoint)))
        self._state_action_csv_path = checkpoint_dir / "rl_policy_dual_state_action.csv"
        with open(self._state_action_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["episode_idx", "rollout_step", "time"]
                + [f"state_{idx}" for idx in range(STATE_DIM)]
                + [f"action_command_joint_pos_{idx}" for idx in range(ACTION_DIM)]
            )

        self._log_path: Optional[Path] = None
        if self.args.rl_log_tsv:
            self._log_path = checkpoint_dir / "rollout_rl_policy_dual_debug_log.tsv"
            with open(self._log_path, "w", newline="") as f:
                writer = csv.writer(f, delimiter="\t")
                writer.writerow(["step_idx", "obs", "raw_action", "direct_joint_command"])

        print(f"[{self.__class__.__name__}] Load PPO checkpoint on {self.device}")
        print(f"[{self.__class__.__name__}] Save state/action CSV: {self._state_action_csv_path}")

    def _get_joint_limits(self):
        action_space = getattr(self.env, "action_space", None)
        if action_space is None:
            raise RuntimeError("Environment must expose action_space.")
        low = torch.as_tensor(action_space.low, dtype=torch.float32, device=self.device).reshape(-1)
        high = torch.as_tensor(action_space.high, dtype=torch.float32, device=self.device).reshape(-1)
        if low.numel() != ACTION_DIM or high.numel() != ACTION_DIM:
            raise ValueError(
                f"[{self.__class__.__name__}] action_space must have dim {ACTION_DIM}, "
                f"got low={low.numel()}, high={high.numel()}."
            )

        low = low.clone()
        high = high.clone()
        for idx in GRIPPER_JOINT_IDX:
            gripper_limits = sorted(
                [
                    gripper_q_robomanip_to_maniskill(float(low[idx].item())),
                    gripper_q_robomanip_to_maniskill(float(high[idx].item())),
                ]
            )
            low[idx] = gripper_limits[0]
            high[idx] = gripper_limits[1]
        return low, high

    def _get_action_delta_limits(self):
        delta = torch.full(
            (ACTION_DIM,),
            ARM_JOINT_DELTA_LIMIT,
            dtype=torch.float32,
            device=self.device,
        )
        delta[GRIPPER_JOINT_IDX] = GRIPPER_JOINT_DELTA_LIMIT
        return -delta, delta

    def setup_variables(self):
        super().setup_variables()
        self.camera_names = []

    def setup_plot(self):
        fig_ax = plt.subplots(2, 1, figsize=(13.5, 6.0), dpi=60, squeeze=False)
        super().setup_plot(fig_ax)

    def reset_variables(self):
        super().reset_variables()
        self.policy_action_buf = None

    def get_state(self):
        qpos = np.asarray(
            self.motion_manager.get_data(DataKey.MEASURED_JOINT_POS, self.obs),
            dtype=np.float32,
        ).reshape(-1)
        qvel = np.asarray(
            self.motion_manager.get_data(DataKey.MEASURED_JOINT_VEL, self.obs),
            dtype=np.float32,
        ).reshape(-1)
        if qpos.size != ACTION_DIM:
            raise ValueError(f"measured joint pos must have dim {ACTION_DIM}, got {qpos.size}.")
        if qvel.size != ACTION_DIM:
            raise ValueError(f"measured joint vel must have dim {ACTION_DIM}, got {qvel.size}.")

        qpos_ms = qpos.copy()
        qvel_ms = qvel.copy()
        for idx in GRIPPER_JOINT_IDX:
            qpos_ms[idx] = gripper_q_robomanip_to_maniskill(float(qpos_ms[idx]))
            qvel_ms[idx] = gripper_qvel_robomanip_to_maniskill(float(qvel_ms[idx]))

        self._latest_joint_pos_tensor = torch.as_tensor(
            qpos_ms, dtype=torch.float32, device=self.device
        )
        state = np.concatenate(
            [
                qpos_ms[LEFT_JOINT_IDX],
                qvel_ms[LEFT_JOINT_IDX],
                qpos_ms[RIGHT_JOINT_IDX],
                qvel_ms[RIGHT_JOINT_IDX],
            ]
        ).astype(np.float32)
        assert state.size == STATE_DIM
        self.state_for_policy = state
        return torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

    def infer_policy(self):
        if self.policy_action_buf is None or len(self.policy_action_buf) == 0:
            obs_tensor = self.get_state()
            with torch.no_grad():
                raw_action = self.policy.get_action(
                    obs_tensor, deterministic=PPO_DETERMINISTIC
                )

            raw_action = raw_action.squeeze(0)
            clipped_action = torch.clamp(
                raw_action,
                self._normalized_action_low,
                self._normalized_action_high,
            )
            delta_scale = (clipped_action - self._normalized_action_low) / (
                self._normalized_action_high - self._normalized_action_low
            )
            denormalized_delta = self._action_delta_low + delta_scale * (
                self._action_delta_high - self._action_delta_low
            )
            direct_joint_command = self._latest_joint_pos_tensor + denormalized_delta
            direct_joint_command = torch.max(
                torch.min(direct_joint_command, self._joint_position_high),
                self._joint_position_low,
            ).clone()
            for idx in GRIPPER_JOINT_IDX:
                direct_joint_command[idx] = direct_joint_command[idx].new_tensor(
                    gripper_q_maniskill_to_robomanip(
                        float(direct_joint_command[idx].item())
                    )
                )

            physical_np = direct_joint_command.detach().cpu().numpy().astype(np.float64)
            physical_np[GRIPPER_JOINT_IDX] = FIXED_GRIPPER_COMMAND
            self._append_state_action_csv(physical_np)
            if self._log_path is not None:
                with open(self._log_path, "a", newline="") as f:
                    writer = csv.writer(f, delimiter="\t")
                    writer.writerow(
                        [
                            int(getattr(self, "rollout_time_idx", 0)),
                            json.dumps(self.state_for_policy.astype(float).tolist()),
                            json.dumps(raw_action.detach().cpu().numpy().astype(float).tolist()),
                            json.dumps(physical_np.astype(float).tolist()),
                        ]
                    )
            self.policy_action_buf = [physical_np]

        self.policy_action = denormalize_data(
            self.policy_action_buf.pop(0), self.model_meta_info["action"]
        )
        self.policy_action_list = np.concatenate(
            [self.policy_action_list, self.policy_action[np.newaxis]]
        )

    def _append_state_action_csv(self, action: np.ndarray):
        rollout_step = int(getattr(self, "rollout_time_idx", 0))
        episode_idx = int(getattr(self.data_manager, "episode_idx", 0))
        elapsed_time = self.phase_manager.phase.get_elapsed_duration()
        with open(self._state_action_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [episode_idx, rollout_step, float(elapsed_time)]
                + self.state_for_policy.astype(float).tolist()
                + np.asarray(action, dtype=float).reshape(-1).tolist()
            )

    def set_command_data(self, action_keys=None):
        super().set_command_data(action_keys)

    def draw_plot(self):
        for _ax in np.ravel(self.ax):
            _ax.cla()
            _ax.axis("off")
        self.plot_action(self.ax[1, 0])
        self.canvas.draw()
        cv2.imshow(
            self.policy_name,
            cv2.cvtColor(np.asarray(self.canvas.buffer_rgba()), cv2.COLOR_RGB2BGR),
        )
