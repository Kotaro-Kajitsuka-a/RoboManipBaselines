import csv
import importlib
import json
import os
import sys
import time
import types
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import cv2
import matplotlib.pylab as plt
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from robo_manip_baselines.common import (
    ArmConfig,
    DataKey,
    RolloutBase,
    denormalize_data,
    normalize_data,
)

from .gripper_utils import (
    gripper_q_maniskill_to_robomanip,
    gripper_q_robomanip_to_maniskill,
    gripper_qvel_robomanip_to_maniskill,
)
from .state_utils import get_adjusted_measured_eef_pose

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class ManiSkillPpoAgent(nn.Module):
    """Reproduction of ManiSkill PPO agent architecture for rollout."""

    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)

        self.critic = nn.Sequential(
            _layer_init(nn.Linear(self.obs_dim, 256)),
            nn.Tanh(),
            _layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            _layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            _layer_init(nn.Linear(256, 1)),
        )
        self.actor_mean = nn.Sequential(
            _layer_init(nn.Linear(self.obs_dim, 256)),
            nn.Tanh(),
            _layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            _layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            _layer_init(nn.Linear(256, self.action_dim), std=0.01 * np.sqrt(2)),
        )
        self.actor_logstd = nn.Parameter(torch.ones(1, self.action_dim) * -0.5)

    def get_value(self, obs):
        return self.critic(obs)

    def get_action(self, obs, deterministic=False):
        action_mean = self.actor_mean(obs)
        if deterministic:
            return action_mean

        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        return probs.sample()


_NORMALIZED_ACTION_LOW = torch.tensor(-1.0, dtype=torch.float32)
_NORMALIZED_ACTION_HIGH = torch.tensor(1.0, dtype=torch.float32)
_DEFAULT_ARM_JOINT_DELTA_LIMIT = 0.03
_DEFAULT_GRIPPER_JOINT_DELTA_LIMIT = 0.1
_PPO_DETERMINISTIC = True
_PPO_USE_CUDA = torch.cuda.is_available()
_PPO_LOG_TSV = True
_PPO_PROFILE = True

class RolloutPpoCus(RolloutBase):
    def run(self):
        return super().run()


    def setup_model_meta_info(self):
        checkpoint_dir = os.path.split(self.args.checkpoint)[0]
        model_meta_info_path = os.path.join(checkpoint_dir, "model_meta_info.pkl")

        if not os.path.isfile(model_meta_info_path):
            raise FileNotFoundError(
                f"[{self.__class__.__name__}] Required model_meta_info.pkl was not found "
                f"next to the checkpoint: {model_meta_info_path}  "
                "Generate the file with CreatePpoCusMetaInfo.py) and re-run."
            )

        super().setup_model_meta_info()
        self._init_joint_metadata()
        self.extra_state_keys: list[str] = []
        self.extra_state_dims: Dict[str, int] = {}
        self.ppo_task_handler = None
        self.ppo_task_params: Dict[str, Any] = {}

        # Backward-compat attribute for tasks that may expect it; left empty.
        self.marker_transform_cache: Dict[int, np.ndarray] = {}
        self._setup_ppo_task_from_meta()

    def _init_joint_metadata(self) -> None:
        """Collect joint index information from the environment for scaling."""
        self._gripper_joint_indices = np.array([], dtype=np.int64)
        
        self._eef_joint_index_map: Dict[int, np.ndarray] = {}
        env = self.env.unwrapped if hasattr(self.env, "unwrapped") else self.env

        command_dim = DataKey.get_dim(DataKey.COMMAND_JOINT_POS, self.env)
        if command_dim != self.action_dim:
            raise ValueError(
                f"[{self.__class__.__name__}] action dim mismatch between meta ({self.action_dim}) "
                f"and env command dim ({command_dim})."
            )

        default_target = np.zeros(self.action_dim, dtype=np.float32)
        body_configs = getattr(env, "body_config_list", None) or []
        gripper_idxes: List[np.ndarray] = []
        for body_config in body_configs:
            if not isinstance(body_config, ArmConfig):
                continue
            if body_config.arm_joint_idxes.size:
                target_arm = np.asarray(body_config.init_arm_joint_pos, dtype=np.float32).reshape(-1)
                default_target[body_config.arm_joint_idxes] = target_arm[: body_config.arm_joint_idxes.size]
            if body_config.gripper_joint_idxes.size:
                target_gripper = np.asarray(
                    body_config.init_gripper_joint_pos, dtype=np.float32
                ).reshape(-1)
                default_target[body_config.gripper_joint_idxes] = target_gripper[
                    : body_config.gripper_joint_idxes.size
                ]
                gripper_idxes.append(body_config.gripper_joint_idxes)
            if body_config.eef_idx is not None:
                joint_idxes = body_config.arm_joint_idxes
                if body_config.gripper_joint_idxes.size:
                    joint_idxes = np.concatenate(
                        [joint_idxes, body_config.gripper_joint_idxes]
                    )
                # Keep track of each arm's joint slice so left/right-only keys can reuse the converted vectors.
                self._eef_joint_index_map[int(body_config.eef_idx)] = joint_idxes.astype(
                    np.int64, copy=False
                )

        if gripper_idxes:
            self._gripper_joint_indices = (
                np.unique(np.concatenate(gripper_idxes).astype(np.int64))
                if len(gripper_idxes) > 1
                else gripper_idxes[0].astype(np.int64)
            )
        self.default_target_joint_pos = default_target

    def _init_action_scaling_tensors(self) -> None:
        """Initialize tensors describing per-joint min/max and delta limits."""
        action_space = getattr(self.env, "action_space", None)
        if action_space is None or not hasattr(action_space, "low") or not hasattr(
            action_space, "high"
        ):
            raise RuntimeError(
                f"[{self.__class__.__name__}] Environment does not expose a continuous action space."
            )

        env_low = torch.as_tensor(
            action_space.low, dtype=torch.float32, device=self.device
        ).reshape(-1)
        env_high = torch.as_tensor(
            action_space.high, dtype=torch.float32, device=self.device
        ).reshape(-1)
        if env_low.numel() != self.action_dim or env_high.numel() != self.action_dim:
            raise ValueError(
                f"[{self.__class__.__name__}] action space dimension mismatch: "
                f"space={env_low.numel()}, meta={self.action_dim}."
            )
        self._joint_position_low = env_low
        self._joint_position_high = env_high

        delta_limit = torch.full(
            (self.action_dim,), _DEFAULT_ARM_JOINT_DELTA_LIMIT, dtype=torch.float32, device=self.device
        )
        if getattr(self, "_gripper_joint_indices", None) is not None and self._gripper_joint_indices.size > 0:
            gripper_idx_tensor = torch.as_tensor(
                self._gripper_joint_indices, dtype=torch.long, device=self.device
            )
            delta_limit[gripper_idx_tensor] = _DEFAULT_GRIPPER_JOINT_DELTA_LIMIT
        self._action_delta_low = -delta_limit
        self._action_delta_high = delta_limit

    def _convert_gripper_positions_to_maniskill(self, values: np.ndarray) -> np.ndarray:
        if getattr(self, "_gripper_joint_indices", None) is None or self._gripper_joint_indices.size == 0:
            return values
        converted = values.copy()
        for idx in self._gripper_joint_indices:
            converted[idx] = gripper_q_robomanip_to_maniskill(float(converted[idx]))
        return converted

    def _convert_gripper_velocities_to_maniskill(self, values: np.ndarray) -> np.ndarray:
        if getattr(self, "_gripper_joint_indices", None) is None or self._gripper_joint_indices.size == 0:
            return values
        converted = values.copy()
        for idx in self._gripper_joint_indices:
            converted[idx] = gripper_qvel_robomanip_to_maniskill(float(converted[idx]))
        return converted

    def _convert_gripper_tensor_to_robomanip(self, tensor: torch.Tensor) -> torch.Tensor:
        if getattr(self, "_gripper_joint_indices", None) is None or self._gripper_joint_indices.size == 0:
            return tensor
        for idx in self._gripper_joint_indices:
            tensor[idx] = tensor[idx].new_tensor(
                gripper_q_maniskill_to_robomanip(float(tensor[idx].item()))
            )
        return tensor

    def _setup_ppo_task_from_meta(self) -> None:
        ppo_task_cfg = self.model_meta_info.get("ppo_task")
        if not ppo_task_cfg:
            return

        extra_keys_cfg = ppo_task_cfg.get("extra_keys", [])
        if not isinstance(extra_keys_cfg, list):
            raise TypeError(
                f"[{self.__class__.__name__}] 'ppo_task.extra_keys' must be a list."
            )

        extra_state_keys: list[str] = []
        extra_state_dims: Dict[str, int] = {}
        for entry in extra_keys_cfg:
            if not isinstance(entry, dict):
                raise TypeError(
                    f"[{self.__class__.__name__}] 'ppo_task.extra_keys' entries must be objects."
                )
            name = entry.get("name")
            dim = entry.get("dim")
            if not name or not isinstance(name, str):
                raise ValueError(
                    f"[{self.__class__.__name__}] Invalid extra key name: {entry}"
                )
            if name in extra_state_keys:
                raise ValueError(
                    f"[{self.__class__.__name__}] Duplicate extra key detected: {name}"
                )
            if dim is None:
                raise ValueError(
                    f"[{self.__class__.__name__}] Missing dimension for extra key '{name}'."
                )
            dim_int = int(dim)
            if dim_int <= 0:
                raise ValueError(
                    f"[{self.__class__.__name__}] Dimension for extra key '{name}' must be positive."
                )
            extra_state_keys.append(name)
            extra_state_dims[name] = dim_int

        missing_in_state = [key for key in extra_state_keys if key not in self.state_keys]
        if missing_in_state:
            raise ValueError(
                f"[{self.__class__.__name__}] Extra state keys {missing_in_state} "
                "are not present in model_meta_info['state']['keys']."
            )

        module_path = ppo_task_cfg.get("module")
        if not module_path or not isinstance(module_path, str):
            raise ValueError(
                f"[{self.__class__.__name__}] 'ppo_task.module' must be a non-empty string."
            )

        try:
            task_module = importlib.import_module(module_path)
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                f"[{self.__class__.__name__}] Failed to import PPO task module '{module_path}'."
            ) from exc

        builder = getattr(task_module, "build_ppo_task", None)
        if builder is None:
            raise AttributeError(
                f"[{self.__class__.__name__}] Module '{module_path}' does not expose a 'build_ppo_task' function."
            )

        params = ppo_task_cfg.get("params") or {}
        if not isinstance(params, dict):
            raise TypeError(
                f"[{self.__class__.__name__}] 'ppo_task.params' must be a dictionary."
            )
        self.ppo_task_params = dict(params)

        self.extra_state_keys = extra_state_keys
        self.extra_state_dims = extra_state_dims
        self.standard_state_keys = [
            key for key in self.state_keys if key not in self.extra_state_keys
        ]

        self.ppo_task_handler = builder(self, self.ppo_task_params)
        if self.ppo_task_handler is None:
            raise RuntimeError(
                f"[{self.__class__.__name__}] Task builder '{module_path}.build_ppo_task' returned None."
            )

        task_name = ppo_task_cfg.get("name") or module_path
        print(
            f"[{self.__class__.__name__}] Loaded PPO task '{task_name}'. "
            f"Extra state keys: {self.extra_state_keys}"
        )

    def setup_policy(self):
        self.print_policy_info()
        print(
            f"  - obs steps: {self.model_meta_info['data']['n_obs_steps']}, action steps: {self.model_meta_info['data']['n_action_steps']}"
        )

        state_dict = torch.load(self.args.checkpoint, map_location="cpu")

        if "actor_mean.0.weight" not in state_dict or "actor_logstd" not in state_dict:
            raise KeyError(
                f"[{self.__class__.__name__}] ManiSkill PPO checkpoint does not contain expected keys."
            )

        obs_dim = state_dict["actor_mean.0.weight"].shape[1]
        action_dim_from_ckpt = int(state_dict["actor_logstd"].shape[-1])
        if action_dim_from_ckpt != self.action_dim:
            raise ValueError(
                f"[{self.__class__.__name__}] action dim mismatch: meta={self.action_dim}, checkpoint={action_dim_from_ckpt}"
            )

        self.policy = ManiSkillPpoAgent(obs_dim, self.action_dim)
        self.policy.load_state_dict(state_dict)

        use_cuda = _PPO_USE_CUDA and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.policy.to(self.device)
        self.policy.eval()
        self._policy_obs_dim = obs_dim

        self._normalized_action_low = torch.full(
            (self.action_dim,), float(_NORMALIZED_ACTION_LOW.item()), device=self.device
        )
        self._normalized_action_high = torch.full(
            (self.action_dim,), float(_NORMALIZED_ACTION_HIGH.item()), device=self.device
        )
        self._init_action_scaling_tensors()

        print(
            f"[{self.__class__.__name__}] Load ManiSkill PPO checkpoint on {self.device}"
        )

        expanded_path = _REPO_ROOT / "robo_manip_baselines" / "rollout_debug_log_expanded.csv"
        header = ["step_idx"]
        header += [f"obs_{idx}" for idx in range(self._policy_obs_dim)]
        header += [f"raw_action_{idx}" for idx in range(self.action_dim)]
        header += [f"direct_joint_command_{idx}" for idx in range(self.action_dim)]
        with open(expanded_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
        self._expanded_csv_path: Optional[Path] = expanded_path
        print(
            f"[{self.__class__.__name__}] Logging expanded observations/actions to {expanded_path}"
        )

        self._log_path = None
        if _PPO_LOG_TSV:
            checkpoint_dir = os.path.dirname(os.path.abspath(self.args.checkpoint))
            default_name = f"{self.__class__.__name__.lower()}_debug_log.tsv"
            self._log_path = os.path.join(checkpoint_dir, default_name)
            os.makedirs(os.path.dirname(self._log_path), exist_ok=True)
            with open(self._log_path, "w", newline="") as f:
                writer = csv.writer(f, delimiter="\t")
                writer.writerow(["step_idx", "obs", "direct_joint_command"])
            print(
                f"[{self.__class__.__name__}] Logging observations and actions to {self._log_path}"
            )

    def setup_variables(self):
        super().setup_variables()

        # Disable vision/image pipelines for this rollout; box pose is handled in the task.
        self.camera_names = []

        self._profile_enabled = bool(_PPO_PROFILE)
        if self._profile_enabled:
            self._profile_data = defaultdict(list)
            self._wrap_profile_hooks()

    def _wrap_profile_hooks(self):
        env = self.env

        original_step = env.step
        rollout_self = self

        def profiled_step(env_self, action):
            start = time.perf_counter()
            result = original_step(action)
            rollout_self._profile_data["env_step"].append(time.perf_counter() - start)
            return result

        env.step = types.MethodType(profiled_step, env)

        original_record_data = self.record_data

        def profiled_record_data():
            start = time.perf_counter()
            result = original_record_data()
            rollout_self._profile_data["record_data"].append(time.perf_counter() - start)
            return result

        self.record_data = profiled_record_data

    def setup_plot(self):
        num_cols = max(len(self.camera_names), 1)
        fig_ax = plt.subplots(
            2,
            num_cols,
            figsize=(13.5, 6.0),
            dpi=60,
            squeeze=False,
            constrained_layout=True,
        )
        super().setup_plot(fig_ax)

    def reset_variables(self):
        super().reset_variables()

        self.state_buf = None
        self.images_buf = None
        self.policy_action_buf = None

    def get_state(self):
        # Defensive init (e.g., if setup_variables was bypassed)
        if not hasattr(self, "marker_transform_cache"):
            self.marker_transform_cache = {}

        extra_state_arrays: Dict[str, np.ndarray] = {}
        if len(self.state_keys) != 0:
            extra_state_values: Dict[str, np.ndarray] = {}
            if self.ppo_task_handler is not None:
                extra_state_raw = self.ppo_task_handler.get_extra_state() or {}
                if not isinstance(extra_state_raw, dict):
                    raise TypeError(
                        f"[{self.__class__.__name__}] Task handler must return a dict of extra states."
                    )
                extra_state_values = {
                    key: np.asarray(value, dtype=np.float32).reshape(-1)
                    for key, value in extra_state_raw.items()
                }

            # Validate extra state contributions
            for key in self.extra_state_keys:
                if key not in extra_state_values:
                    raise KeyError(
                        f"[{self.__class__.__name__}] Extra state '{key}' missing from PPO task handler output."
                    )
                arr = extra_state_values[key]
                expected_dim = self.extra_state_dims.get(key)
                if expected_dim is not None and arr.size != expected_dim:
                    raise ValueError(
                        f"[{self.__class__.__name__}] Extra state '{key}' has dimension {arr.size}, "
                        f"expected {expected_dim}."
                    )
                extra_state_arrays[key] = arr

        qpos = self.motion_manager.get_data(DataKey.MEASURED_JOINT_POS, self.obs)
        qvel = self.motion_manager.get_data(DataKey.MEASURED_JOINT_VEL, self.obs)

        qpos_ms = self._convert_gripper_positions_to_maniskill(
            qpos.astype(np.float32).copy()
        )
        qvel_ms = self._convert_gripper_velocities_to_maniskill(
            qvel.astype(np.float32).copy()
        )
        # Cache the latest joint position tensor for action denormalization
        self._latest_joint_pos_tensor = torch.as_tensor(
            qpos_ms, dtype=torch.float32, device=self.device
        )

        def _get_joint_slice(key: str) -> Optional[np.ndarray]:
            idx_map = getattr(self, "_eef_joint_index_map", None) or {}
            if key in (
                DataKey.LEFT_MEASURED_JOINT_POS,
                DataKey.LEFT_MEASURED_JOINT_VEL,
            ):
                target_idx = 0
            elif key in (
                DataKey.RIGHT_MEASURED_JOINT_POS,
                DataKey.RIGHT_MEASURED_JOINT_VEL,
            ):
                target_idx = 1
            else:
                return None
            joint_slice = idx_map.get(target_idx)
            if joint_slice is None:
                raise ValueError(
                    f"[{self.__class__.__name__}] State key '{key}' requires joint slice for eef_idx={target_idx}, "
                    "but no matching arm configuration was found."
                )
            return joint_slice

        def _component_for_key(state_key: str) -> np.ndarray:
            # Always pull from the ManiSkill-aligned joint vectors to keep scaling consistent with training.
            if state_key in extra_state_arrays:
                return extra_state_arrays[state_key]
            if state_key == DataKey.MEASURED_JOINT_POS:
                return qpos_ms
            if state_key == DataKey.MEASURED_JOINT_VEL:
                return qvel_ms
            if state_key in (
                DataKey.LEFT_MEASURED_JOINT_POS,
                DataKey.RIGHT_MEASURED_JOINT_POS,
            ):
                indices = _get_joint_slice(state_key)
                return qpos_ms[indices]
            if state_key in (
                DataKey.LEFT_MEASURED_JOINT_VEL,
                DataKey.RIGHT_MEASURED_JOINT_VEL,
            ):
                indices = _get_joint_slice(state_key)
                return qvel_ms[indices]
            return self.motion_manager.get_data(state_key, self.obs)


        eef = self.motion_manager.get_data(DataKey.MEASURED_EEF_POSE, self.obs)
        left, right = get_adjusted_measured_eef_pose(eef)



        if len(self.state_keys) == 0:
            state_vector = np.zeros(0, dtype=np.float32)
        else:
            components: List[np.ndarray] = []
            for state_key in self.state_keys:
                component = np.asarray(
                    _component_for_key(state_key), dtype=np.float32
                ).reshape(-1)
                components.append(component)

            state_vector = (
                np.concatenate(components).astype(np.float32)
                if components
                else np.zeros(0, dtype=np.float32)
            )
            expected_state_dim = len(self.model_meta_info["state"]["example"])
            if state_vector.size != expected_state_dim:
                raise ValueError(
                    f"[{self.__class__.__name__}] State dimension mismatch. "
                    f"Constructed {state_vector.size} elements, "
                    f"but model_meta_info expects {expected_state_dim}."
                )

        self.state_for_ppo = state_vector.copy()

        norm_state = normalize_data(state_vector, self.model_meta_info["state"])

        state = torch.tensor(norm_state, dtype=torch.float32)

        # Store and return
        if self.state_buf is None:
            self.state_buf = [
                state for _ in range(self.model_meta_info["data"]["n_obs_steps"])
            ]
        else:
            self.state_buf.pop(0)
            self.state_buf.append(state)

        state = torch.stack(self.state_buf, dim=0)[torch.newaxis].to(self.device)

        return state

    def get_images(self):
        if len(self.camera_names) == 0:
            return None

        images = []
        for camera_name in self.camera_names:
            rgb_image = self.info["rgb_images"][camera_name]

            image = np.moveaxis(rgb_image, -1, -3)
            image = torch.tensor(image.copy(), dtype=torch.uint8)
            image = self.image_transforms(image)

            images.append(image)

        # Store and return
        if self.images_buf is None:
            self.images_buf = [
                [image for _ in range(self.model_meta_info["data"]["n_obs_steps"])]
                for image in images
            ]
        else:
            for single_images_buf, image in zip(self.images_buf, images):
                single_images_buf.pop(0)
                single_images_buf.append(image)

        images = torch.stack(
            [
                torch.stack(single_images_buf, dim=0)[torch.newaxis].to(self.device)
                for single_images_buf in self.images_buf
            ]
        )

        return images

    def infer_policy(self):
        # Infer
        if self.policy_action_buf is None or len(self.policy_action_buf) == 0:
            profile_enabled = getattr(self, "_profile_enabled", False)
            if profile_enabled:
                timer = time.perf_counter
                total_start = timer()
                state_start = timer()

            self.get_state()  # update buffers and logs

            if profile_enabled:
                self._profile_data["state_fetch"].append(timer() - state_start)

            
            obs_tensor = torch.tensor(
                self.state_for_ppo, dtype=torch.float32, device=self.device
            ).unsqueeze(0)

            if profile_enabled:
                policy_start = timer()

            with torch.no_grad():
                raw_action = self.policy.get_action(
                    obs_tensor, deterministic=_PPO_DETERMINISTIC
                )

            if profile_enabled:
                self._profile_data["policy_forward"].append(timer() - policy_start)

            raw_action = raw_action.squeeze(0)
            clipped_action = torch.clamp(
                raw_action, self._normalized_action_low, self._normalized_action_high
            )

            if clipped_action.numel() != self.action_dim:
                raise ValueError(
                    f"[{self.__class__.__name__}] Unexpected action length {clipped_action.numel()} "
                    f"for action_dim {self.action_dim}."
                )

            normalized_span = self._normalized_action_high - self._normalized_action_low
            delta_scale = (clipped_action - self._normalized_action_low) / normalized_span
            denormalized_delta = self._action_delta_low + delta_scale * (
                self._action_delta_high - self._action_delta_low
            )

            # Use the cached measured joint positions (not the reordered obs) for delta application
            current_joint_pos = getattr(self, "_latest_joint_pos_tensor", None)
            if current_joint_pos is None:
                current_joint_pos = obs_tensor[..., : self.action_dim].squeeze(0)
            direct_joint_command = current_joint_pos + denormalized_delta
            direct_joint_command = torch.max(
                torch.min(direct_joint_command, self._joint_position_high),
                self._joint_position_low,
            )

            if direct_joint_command.numel() > 0:
                direct_joint_command = direct_joint_command.clone()
                direct_joint_command = self._convert_gripper_tensor_to_robomanip(
                    direct_joint_command
                )

            physical_np = direct_joint_command.detach().cpu().numpy().astype(np.float64)
            if getattr(self, "_gripper_joint_indices", None) is not None and self._gripper_joint_indices.size > 0:
                physical_np[self._gripper_joint_indices] = 119.0

            if hasattr(self, "_log_path") and self._log_path:
                obs_list = (
                    obs_tensor.squeeze(0).detach().cpu().numpy().astype(np.float64).tolist()
                )
                direct_list = physical_np.tolist()
                with open(self._log_path, "a", newline="") as f:
                    writer = csv.writer(f, delimiter="\t")
                    writer.writerow(
                        [
                            int(getattr(self, "rollout_time_idx", 0)),
                            json.dumps(obs_list),
                            json.dumps(direct_list),
                        ]
                    )
            if self._expanded_csv_path is not None:
                obs_values = obs_tensor.squeeze(0).detach().cpu().numpy().astype(np.float64)
                obs_values = obs_values[: self._policy_obs_dim]
                if obs_values.size < self._policy_obs_dim:
                    obs_values = np.pad(
                        obs_values,
                        (0, self._policy_obs_dim - obs_values.size),
                        mode="constant",
                        constant_values=0.0,
                    )
                raw_action_values = raw_action.detach().cpu().numpy().astype(np.float64).tolist()
                if len(raw_action_values) < self.action_dim:
                    raw_action_values += [0.0] * (self.action_dim - len(raw_action_values))

                row = [int(getattr(self, "rollout_time_idx", 0))]
                row += obs_values.tolist()
                row += raw_action_values[: self.action_dim]
                action_values = physical_np.tolist()
                if len(action_values) < self.action_dim:
                    action_values += [0.0] * (self.action_dim - len(action_values))
                row += action_values[: self.action_dim]
                with open(self._expanded_csv_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(row)

            self.policy_action_buf = [physical_np]

            if profile_enabled:
                self._profile_data["infer_total"].append(timer() - total_start)

        # Store action
        self.policy_action = denormalize_data(
            self.policy_action_buf.pop(0), self.model_meta_info["action"]
        )
        self.policy_action_list = np.concatenate(
            [self.policy_action_list, self.policy_action[np.newaxis]]
        )

    def reset(self):
        super().reset()
        if self.ppo_task_handler and hasattr(self.ppo_task_handler, "on_reset"):
            self.ppo_task_handler.on_reset()

    def set_command_data(self, action_keys=None):
        if getattr(self, "_profile_enabled", False):
            start = time.perf_counter()
            super().set_command_data(action_keys)
            self._profile_data["set_command_data"].append(time.perf_counter() - start)
        else:
            super().set_command_data(action_keys)

    def draw_plot(self):
        # Clear plot
        for _ax in np.ravel(self.ax):
            _ax.cla()
            _ax.axis("off")

        # Plot images
        self.plot_images(self.ax[0, 0 : len(self.camera_names)])

        # Plot action
        self.plot_action(self.ax[1, 0])

        # Finalize plot
        self.canvas.draw()
        cv2.imshow(
            self.policy_name,
            cv2.cvtColor(np.asarray(self.canvas.buffer_rgba()), cv2.COLOR_RGB2BGR),
        )

    def print_statistics(self):
        super().print_statistics()
        if getattr(self, "_profile_enabled", False) and self._profile_data:
            print(f"[{self.__class__.__name__}] Profiling summary")
            for key, samples in self._profile_data.items():
                if not samples:
                    continue
                samples_arr = np.array(samples)
                print(
                    f"  - {key} [s] | mean: {samples_arr.mean():.2e}, "
                    f"std: {samples_arr.std():.2e}, min: {samples_arr.min():.2e}, max: {samples_arr.max():.2e}"
                )
