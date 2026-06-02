from __future__ import annotations

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
)

EXPECTED_STATE_KEYS = [
    "agent/qpos",
    "agent/qvel",
    "extra/inner_marker_panel_position",
    "extra/inner_marker_panel_rotation_6d",
    "extra/outer_marker_panel_position",
    "extra/outer_marker_panel_rotation_6d",
]
EXPECTED_STATE_DIM = 33
EXPECTED_ACTION_DIM = 8
ARM_JOINT_DELTA_LIMIT = 0.03
GRIPPER_JOINT_DELTA_LIMIT = 0.01
STATE_COMPONENTS = [
    ("agent/qpos", 8),
    ("agent/qvel", 7),
    ("extra/inner_marker_panel_position", 3),
    ("extra/inner_marker_panel_rotation_6d", 6),
    ("extra/outer_marker_panel_position", 3),
    ("extra/outer_marker_panel_rotation_6d", 6),
]

_NORMALIZED_ACTION_LOW = torch.tensor(-1.0, dtype=torch.float32)
_NORMALIZED_ACTION_HIGH = torch.tensor(1.0, dtype=torch.float32)
_PPO_DETERMINISTIC = True


def _layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def _state_csv_headers():
    headers = []
    for name, dim in STATE_COMPONENTS:
        safe_name = name.replace("/", "_")
        headers.extend([f"state_{safe_name}_{idx}" for idx in range(dim)])
    return headers


def _action_csv_headers():
    return [f"action_command_joint_pos_{idx}" for idx in range(EXPECTED_ACTION_DIM)]


class ManiSkillPpoAgent(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int):
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


class RolloutRLPolicy(RolloutBase):
    """Single xArm7 rollout for the ManiSkill cardboard-cabinet PPO policy."""

    def set_additional_args(self, parser):
        parser.add_argument(
            "--rl-log-tsv",
            action="store_true",
            help="Log observations and actions next to the checkpoint.",
        )

    def setup_model_meta_info(self):
        super().setup_model_meta_info()
        self._validate_meta_info()
        self.extra_state_keys = EXPECTED_STATE_KEYS[2:]
        self.extra_state_dims = {
            "extra/inner_marker_panel_position": 3,
            "extra/inner_marker_panel_rotation_6d": 6,
            "extra/outer_marker_panel_position": 3,
            "extra/outer_marker_panel_rotation_6d": 6,
        }
        self.rl_task_handler = None
        self._setup_cabinet_task()

    def _validate_meta_info(self):
        if list(self.state_keys) != EXPECTED_STATE_KEYS:
            raise ValueError(
                f"[{self.__class__.__name__}] state keys must be exactly "
                f"{EXPECTED_STATE_KEYS}, got {self.state_keys}"
            )
        if self.state_dim != EXPECTED_STATE_DIM:
            raise ValueError(
                f"[{self.__class__.__name__}] state dim must be {EXPECTED_STATE_DIM}, got {self.state_dim}"
            )
        if self.action_dim != EXPECTED_ACTION_DIM:
            raise ValueError(
                f"[{self.__class__.__name__}] action dim must be {EXPECTED_ACTION_DIM}, got {self.action_dim}"
            )
        if list(self.action_keys) != [DataKey.COMMAND_JOINT_POS]:
            raise ValueError(
                f"[{self.__class__.__name__}] action keys must be {[DataKey.COMMAND_JOINT_POS]}, "
                f"got {self.action_keys}"
            )

    def _setup_cabinet_task(self):
        from .rl_tasks.single_cardboard_cabinet import build_rl_task

        task_cfg = self.model_meta_info.get("rl_task") or {}
        params = task_cfg.get("params", {}) if isinstance(task_cfg, dict) else {}
        if not isinstance(params, dict):
            raise TypeError("model_meta_info['rl_task']['params'] must be a dict.")
        self.rl_task_handler = build_rl_task(self, params)

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
        if obs_dim != EXPECTED_STATE_DIM or action_dim != EXPECTED_ACTION_DIM:
            raise ValueError(
                f"[{self.__class__.__name__}] checkpoint dims must be obs={EXPECTED_STATE_DIM}, "
                f"action={EXPECTED_ACTION_DIM}; got obs={obs_dim}, action={action_dim}."
            )

        self.policy = ManiSkillPpoAgent(obs_dim, action_dim)
        self.policy.load_state_dict(state_dict)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy.to(self.device)
        self.policy.eval()

        self._normalized_action_low = torch.full(
            (EXPECTED_ACTION_DIM,),
            float(_NORMALIZED_ACTION_LOW.item()),
            device=self.device,
        )
        self._normalized_action_high = torch.full(
            (EXPECTED_ACTION_DIM,),
            float(_NORMALIZED_ACTION_HIGH.item()),
            device=self.device,
        )
        self._physical_joint_position_low, self._physical_joint_position_high = (
            self._get_physical_joint_limits()
        )
        self._policy_joint_position_low, self._policy_joint_position_high = (
            self._get_policy_joint_limits()
        )
        self._action_delta_low, self._action_delta_high = (
            self._get_action_delta_limits()
        )

        checkpoint_dir = Path(os.path.dirname(os.path.abspath(self.args.checkpoint)))
        self._state_action_csv_path = checkpoint_dir / "rl_policy_state_action.csv"
        with open(self._state_action_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["episode_idx", "rollout_step", "time"]
                + _state_csv_headers()
                + _action_csv_headers()
            )

        self._log_path: Optional[Path] = None
        if self.args.rl_log_tsv:
            self._log_path = checkpoint_dir / (
                f"{self.__class__.__name__.lower()}_debug_log.tsv"
            )
            with open(self._log_path, "w", newline="") as f:
                writer = csv.writer(f, delimiter="\t")
                writer.writerow(
                    ["step_idx", "obs", "raw_action", "direct_joint_command"]
                )

        print(
            f"[{self.__class__.__name__}] Load ManiSkill PPO checkpoint on {self.device}"
        )
        print(
            f"[{self.__class__.__name__}] Save state/action CSV: {self._state_action_csv_path}"
        )

    def _get_physical_joint_limits(self):
        action_space = getattr(self.env, "action_space", None)
        if (
            action_space is None
            or not hasattr(action_space, "low")
            or not hasattr(action_space, "high")
        ):
            raise RuntimeError("Environment must expose a continuous action_space.")
        low = torch.as_tensor(
            action_space.low, dtype=torch.float32, device=self.device
        ).reshape(-1)
        high = torch.as_tensor(
            action_space.high, dtype=torch.float32, device=self.device
        ).reshape(-1)
        if low.numel() != EXPECTED_ACTION_DIM or high.numel() != EXPECTED_ACTION_DIM:
            raise ValueError(
                f"[{self.__class__.__name__}] action_space must have dim {EXPECTED_ACTION_DIM}, "
                f"got low={low.numel()}, high={high.numel()}."
            )
        return low, high

    def _get_policy_joint_limits(self):
        low = self._physical_joint_position_low.clone()
        high = self._physical_joint_position_high.clone()
        gripper_limits = sorted(
            [
                gripper_q_robomanip_to_maniskill(float(low[-1].item())),
                gripper_q_robomanip_to_maniskill(float(high[-1].item())),
            ]
        )
        low[-1] = gripper_limits[0]
        high[-1] = gripper_limits[1]
        return low, high

    def _get_action_delta_limits(self):
        delta = torch.full(
            (EXPECTED_ACTION_DIM,),
            ARM_JOINT_DELTA_LIMIT,
            dtype=torch.float32,
            device=self.device,
        )
        delta[-1] = GRIPPER_JOINT_DELTA_LIMIT
        return -delta, delta

    def setup_variables(self):
        super().setup_variables()
        self.camera_names = []
        self._detector_camera_name = "front"
        camera_names = list(self.data_manager.meta_data.get("camera_names", []))
        if self._detector_camera_name not in camera_names:
            camera_names.append(self._detector_camera_name)
        self.data_manager.meta_data["camera_names"] = camera_names

    def setup_plot(self):
        fig_ax = plt.subplots(2, 1, figsize=(13.5, 6.0), dpi=60, squeeze=False)
        super().setup_plot(fig_ax)

    def reset_variables(self):
        super().reset_variables()
        self.policy_action_buf = None
        self._record_detector_rgb_last = None
        self._outer_marker_seq_prev = None
        self._outer_marker_stagnation = 0
        self._inner_marker_seq_prev = None
        self._inner_marker_stagnation = 0
        self._image_seq_prev = None
        self._image_stagnation = 0

    @staticmethod
    def _rotation_matrix_to_6d(rotation: np.ndarray) -> np.ndarray:
        x_axis = rotation[:, 0]
        y_axis = rotation[:, 1]
        x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-8)
        y_axis = y_axis - np.dot(x_axis, y_axis) * x_axis
        y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-8)
        return np.concatenate([x_axis, y_axis]).astype(np.float32)

    def get_state(self):
        qpos = self.motion_manager.get_data(DataKey.MEASURED_JOINT_POS, self.obs)
        qvel = self.motion_manager.get_data(DataKey.MEASURED_JOINT_VEL, self.obs)
        qpos = np.asarray(qpos, dtype=np.float32).reshape(-1)
        qvel = np.asarray(qvel, dtype=np.float32).reshape(-1)

        if qpos.size != EXPECTED_ACTION_DIM:
            raise ValueError(f"agent/qpos must have dim 8, got {qpos.size}.")
        if qvel.size != EXPECTED_ACTION_DIM:
            raise ValueError(f"RealXarm7 joint_vel must have dim 8, got {qvel.size}.")

        qpos_ms = qpos.copy()
        qpos_ms[-1] = gripper_q_robomanip_to_maniskill(float(qpos_ms[-1]))
        # RealXarm7 returns arm 7D + gripper 1D velocity; ManiSkill policy uses arm velocity only.
        qvel_arm = qvel[:-1].copy()
        self._latest_joint_pos_tensor = torch.as_tensor(
            qpos_ms, dtype=torch.float32, device=self.device
        )

        extra = self.rl_task_handler.get_extra_state()
        parts = [
            qpos_ms,
            qvel_arm,
            extra["extra/inner_marker_panel_position"],
            extra["extra/inner_marker_panel_rotation_6d"],
            extra["extra/outer_marker_panel_position"],
            extra["extra/outer_marker_panel_rotation_6d"],
        ]
        state = np.concatenate(
            [np.asarray(part, dtype=np.float32).reshape(-1) for part in parts]
        ).astype(np.float32)
        if state.size != EXPECTED_STATE_DIM:
            raise ValueError(
                f"Constructed state dim {state.size}, expected {EXPECTED_STATE_DIM}."
            )
        self.state_for_policy = state
        return torch.as_tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

    def infer_policy(self):
        if self.policy_action_buf is None or len(self.policy_action_buf) == 0:
            obs_tensor = self.get_state()
            with torch.no_grad():
                raw_action = self.policy.get_action(
                    obs_tensor, deterministic=_PPO_DETERMINISTIC
                )

            raw_action = raw_action.squeeze(0)
            clipped_action = torch.clamp(
                raw_action, self._normalized_action_low, self._normalized_action_high
            )
            delta_scale = (clipped_action - self._normalized_action_low) / (
                self._normalized_action_high - self._normalized_action_low
            )
            denormalized_delta = self._action_delta_low + delta_scale * (
                self._action_delta_high - self._action_delta_low
            )
            direct_joint_command = self._latest_joint_pos_tensor + denormalized_delta
            direct_joint_command = torch.max(
                torch.min(direct_joint_command, self._policy_joint_position_high),
                self._policy_joint_position_low,
            ).clone()
            direct_joint_command[-1] = direct_joint_command[-1].new_tensor(
                gripper_q_maniskill_to_robomanip(float(direct_joint_command[-1].item()))
            )
            direct_joint_command = torch.max(
                torch.min(direct_joint_command, self._physical_joint_position_high),
                self._physical_joint_position_low,
            )

            physical_np = direct_joint_command.detach().cpu().numpy().astype(np.float64)
            self._append_state_action_csv(physical_np)
            if self._log_path is not None:
                with open(self._log_path, "a", newline="") as f:
                    writer = csv.writer(f, delimiter="\t")
                    writer.writerow(
                        [
                            int(getattr(self, "rollout_time_idx", 0)),
                            json.dumps(self.state_for_policy.astype(float).tolist()),
                            json.dumps(
                                raw_action.detach().cpu().numpy().astype(float).tolist()
                            ),
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

    @staticmethod
    def _call_optional_handler_method(handler, method_names):
        for method_name in method_names:
            getter = getattr(handler, method_name, None)
            if callable(getter):
                return getter()
        return None

    def _update_stagnation(self, label, seq, prev_seq, stagnation):
        if prev_seq is None:
            prev_seq = seq
            if seq is None:
                stagnation = 1
        elif seq == prev_seq:
            stagnation += 1
        else:
            stagnation = 0
            prev_seq = seq

        if stagnation >= 2:
            print(
                f"[{self.__class__.__name__}] WARNING: {label} stagnated for {stagnation} steps.",
                flush=True,
            )
        if stagnation >= 4:
            raise RuntimeError(
                f"[{self.__class__.__name__}] {label} stalled for {stagnation} steps."
            )
        return prev_seq, stagnation

    def record_data(self):
        super().record_data()
        rgb = self._call_optional_handler_method(
            self.rl_task_handler,
            ["get_latest_front_rgb"],
        )
        if rgb is None:
            if self._record_detector_rgb_last is None:
                self._record_detector_rgb_last = np.zeros((480, 640, 3), dtype=np.uint8)
            rgb = self._record_detector_rgb_last
        else:
            self._record_detector_rgb_last = rgb
        self.data_manager.append_single_data(DataKey.get_rgb_image_key("front"), rgb)

        outer_seq = self.rl_task_handler.get_latest_outer_marker_seq()
        inner_seq = self.rl_task_handler.get_latest_inner_marker_seq()
        image_seq = self.rl_task_handler.get_latest_image_seq()
        self.data_manager.append_single_data(
            "outer_marker_seq", -1 if outer_seq is None else int(outer_seq)
        )
        self.data_manager.append_single_data(
            "inner_marker_seq", -1 if inner_seq is None else int(inner_seq)
        )
        self.data_manager.append_single_data(
            "image_seq", -1 if image_seq is None else int(image_seq)
        )

        self._outer_marker_seq_prev, self._outer_marker_stagnation = (
            self._update_stagnation(
                "outer marker detection",
                outer_seq,
                self._outer_marker_seq_prev,
                self._outer_marker_stagnation,
            )
        )
        self._inner_marker_seq_prev, self._inner_marker_stagnation = (
            self._update_stagnation(
                "inner marker detection",
                inner_seq,
                self._inner_marker_seq_prev,
                self._inner_marker_stagnation,
            )
        )
        self._image_seq_prev, self._image_stagnation = self._update_stagnation(
            "image capture",
            image_seq,
            self._image_seq_prev,
            self._image_stagnation,
        )

    def reset(self):
        super().reset()
        if self.rl_task_handler and hasattr(self.rl_task_handler, "on_reset"):
            self.rl_task_handler.on_reset()

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
