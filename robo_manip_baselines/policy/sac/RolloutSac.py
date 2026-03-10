import importlib
import os
from types import SimpleNamespace
from typing import Any, Dict

import cv2
import matplotlib.pylab as plt
import numpy as np
import torch
from gymnasium.spaces import Box

from robo_manip_baselines.common import (
    DataKey,
    RolloutBase,
    denormalize_data,
)

from .SacPolicy import Actor
from .gripper_utils import (
    convert_gripper_positions_to_maniskill,
    convert_gripper_tensor_to_robomanip,
    convert_gripper_velocities_to_maniskill,
)
from .state_utils import get_adjusted_measured_eef_pose

_NORMALIZED_ACTION_LOW = torch.tensor(-1.0, dtype=torch.float32)
_NORMALIZED_ACTION_HIGH = torch.tensor(1.0, dtype=torch.float32)
_DEFAULT_ARM_JOINT_DELTA_LIMIT = 0.045
_DEFAULT_GRIPPER_JOINT_DELTA_LIMIT = 0.1
_SAC_DETERMINISTIC = True
_SAC_USE_CUDA = torch.cuda.is_available()

class RolloutSac(RolloutBase):
    def set_additional_args(self, parser):
        parser.add_argument(
            "--yolo_pt",
            type=str,
            default=None,
            help="path to YOLO pose .pt (use estimator if provided)",
        )

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
        """Hard-coded joint index metadata for RealXarm7Dual (for readability)."""

        command_dim = DataKey.get_dim(DataKey.COMMAND_JOINT_POS, self.env)
        if command_dim != self.action_dim:
            raise ValueError(
                f"[{self.__class__.__name__}] action dim mismatch between meta ({self.action_dim}) "
                f"and env command dim ({command_dim})."
            )
        if self.action_dim != 16:
            raise ValueError(
                f"[{self.__class__.__name__}] This SAC rollout is hard-coded for RealXarm7Dual (action_dim=16), "
                f"got action_dim={self.action_dim}."
            )

        self._gripper_joint_indices = np.array([7, 15], dtype=np.int64)
        self._joint_indices_by_arm_id = {
            0: np.arange(0, 8, dtype=np.int64),
            1: np.arange(8, 16, dtype=np.int64),
        }

    def _setup_ppo_task_from_meta(self) -> None:
        ppo_task_cfg = self.model_meta_info.get("ppo_task")
        if not ppo_task_cfg:
            return

        extra_keys_cfg = ppo_task_cfg.get("extra_keys", [])
        assert isinstance(extra_keys_cfg, list)

        extra_state_keys: list[str] = []
        extra_state_dims: Dict[str, int] = {}
        for entry in extra_keys_cfg:
            name = entry["name"]
            dim_int = int(entry["dim"])
            assert isinstance(name, str) and name
            assert name not in extra_state_dims
            assert dim_int > 0
            extra_state_keys.append(name)
            extra_state_dims[name] = dim_int

        params = ppo_task_cfg.get("params") or {}
        assert isinstance(params, dict)
        self.ppo_task_params = dict(params)

        module_path = ppo_task_cfg["module"]
        assert isinstance(module_path, str) and module_path

        if getattr(self.args, "yolo_pt", None):
            module_path = "robo_manip_baselines.policy.sac.sac_tasks.box_estimator"
            self.ppo_task_params["pt_path"] = self.args.yolo_pt
            self.ppo_task_params["camera_name"] = "top"

        try:
            task_module = importlib.import_module(module_path)
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                f"[{self.__class__.__name__}] Failed to import PPO task module '{module_path}'."
            ) from exc

        builder = task_module.build_ppo_task

        self.extra_state_keys = extra_state_keys
        self.extra_state_dims = extra_state_dims
        self.ppo_task_handler = builder(self, self.ppo_task_params)
        assert self.ppo_task_handler is not None

        task_name = ppo_task_cfg.get("name") or module_path
        print(
            f"[{self.__class__.__name__}] Loaded PPO task '{task_name}'. "
            f"Extra state keys: {self.extra_state_keys}"
        )

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

    def setup_policy(self):
        self.print_policy_info()
        print(
            f"  - obs steps: {self.model_meta_info['data']['n_obs_steps']}, action steps: {self.model_meta_info['data']['n_action_steps']}"
        )

        raw_ckpt = torch.load(self.args.checkpoint, map_location="cpu")
        state_dict = raw_ckpt["actor"] if isinstance(raw_ckpt, dict) and "actor" in raw_ckpt else raw_ckpt

        policy_env = self._build_policy_env_from_meta_info()
        self.policy = Actor(policy_env)
        self.policy.load_state_dict(state_dict)

        use_cuda = _SAC_USE_CUDA and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.policy.to(self.device)
        self.policy.eval()

        self._normalized_action_low = torch.full(
            (self.action_dim,), float(_NORMALIZED_ACTION_LOW.item()), device=self.device
        )
        self._normalized_action_high = torch.full(
            (self.action_dim,), float(_NORMALIZED_ACTION_HIGH.item()), device=self.device
        )
        self._init_action_scaling_tensors()

        print(
            f"[{self.__class__.__name__}] Load ManiSkill SAC checkpoint on {self.device}"
        )

    def _build_policy_env_from_meta_info(self):
        obs_dim = len(self.model_meta_info["state"]["example"])
        action_dim = len(self.model_meta_info["action"]["example"])
        action_low = self.model_meta_info["action"].get("low")
        action_high = self.model_meta_info["action"].get("high")
        if action_low is None or action_high is None:
            action_low = np.full(action_dim, -1.0, dtype=np.float32)
            action_high = np.full(action_dim, 1.0, dtype=np.float32)
        else:
            action_low = np.asarray(action_low, dtype=np.float32).reshape(-1)
            action_high = np.asarray(action_high, dtype=np.float32).reshape(-1)
        if action_low.size != action_dim or action_high.size != action_dim:
            raise ValueError(
                f"[{self.__class__.__name__}] action bounds mismatch: "
                f"low={action_low.size}, high={action_high.size}, expected={action_dim}."
            )
        return SimpleNamespace(
            single_observation_space=Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
            single_action_space=Box(
                low=action_low, high=action_high, dtype=np.float32
            ),
        )

    def setup_variables(self):
        super().setup_variables()

        # Disable vision/image pipelines for this rollout; box pose is handled in the task.
        self.camera_names = []
        self._detector_camera_name = (
            "top" if getattr(self.args, "yolo_pt", None) else "front"
        )
        # Even if the env does not provide the detector camera (e.g. disabled to avoid
        # RealSense conflicts), we still record the detector-thread image under a fixed
        # camera key in metadata.
        camera_names = list(self.data_manager.meta_data.get("camera_names", []))
        if self._detector_camera_name not in camera_names:
            camera_names.append(self._detector_camera_name)
        self.data_manager.meta_data["camera_names"] = camera_names

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

        self.policy_action_buf = None
        self._record_detector_rgb_last = None
        self._box_pose_seq_prev = None
        self._box_pose_stagnation = 0
        self._image_seq_prev = None
        self._image_stagnation = 0

    @staticmethod
    def _call_optional_handler_method(handler, method_names):
        if handler is None:
            return None
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
        else:
            if seq == prev_seq:
                stagnation += 1
            else:
                stagnation = 0
                prev_seq = seq

        if stagnation >= 2:
            print(
                f"[{self.__class__.__name__}] WARNING: {label} stagnated for "
                f"{stagnation} steps.",
                flush=True,
            )
        if stagnation >= 4:
            raise RuntimeError(
                f"[{self.__class__.__name__}] {label} stalled for {stagnation} steps."
            )

        return prev_seq, stagnation

    def record_data(self):
        """detector thread の RGB フレームを使って rollout 1 step 分のデータを記録する。

        SAC の box task では、別スレッドで独立した RealSense pipeline を動かしている
        (`policy/sac/sac_tasks/*` を参照)。同じ RealSense を env 側の camera pipeline から
        同時に開こうとすると "device busy" で失敗しやすいため、SAC では env camera を無効化している。

        それでも画像を保存できるように、detector thread 側の RGB フレームを
        `<detector_camera_name>_rgb_image` というキーで常に追加する。
        """

        # Record standard signals (time, reward, measured/command/rel, tactile, etc.)
        super().record_data()

        rgb = self._call_optional_handler_method(
            self.ppo_task_handler,
            [
                f"get_latest_{self._detector_camera_name}_rgb",
                "get_latest_front_rgb",
            ],
        )

        if rgb is None:
            if self._record_detector_rgb_last is None:
                self._record_detector_rgb_last = np.zeros((480, 640, 3), dtype=np.uint8)
            rgb = self._record_detector_rgb_last
        else:
            self._record_detector_rgb_last = rgb

        self.data_manager.append_single_data(
            DataKey.get_rgb_image_key(self._detector_camera_name), rgb
        )

        board_corners = self._call_optional_handler_method(
            self.ppo_task_handler,
            [
                f"get_latest_{self._detector_camera_name}_aruco_board_corners",
                "get_latest_front_aruco_board_corners",
            ],
        )
        if board_corners is None:
            board_corners = np.full((4, 2), -1.0, dtype=np.float32)
        self.data_manager.append_single_data(
            f"{self._detector_camera_name}_aruco_board_corners",
            board_corners,
        )

        box_seq = None
        image_seq = None
        if self.ppo_task_handler is not None:
            box_seq_getter = getattr(
                self.ppo_task_handler, "get_latest_box_pose_seq", None
            )
            if callable(box_seq_getter):
                box_seq = box_seq_getter()
            image_seq_getter = getattr(
                self.ppo_task_handler, "get_latest_image_seq", None
            )
            if callable(image_seq_getter):
                image_seq = image_seq_getter()

        self.data_manager.append_single_data(
            "box_pose_seq", -1 if box_seq is None else int(box_seq)
        )
        self.data_manager.append_single_data(
            "image_seq", -1 if image_seq is None else int(image_seq)
        )

        self._box_pose_seq_prev, self._box_pose_stagnation = self._update_stagnation(
            "box pose detection",
            box_seq,
            self._box_pose_seq_prev,
            self._box_pose_stagnation,
        )
        self._image_seq_prev, self._image_stagnation = self._update_stagnation(
            "image capture",
            image_seq,
            self._image_seq_prev,
            self._image_stagnation,
        )

    def _get_arm_joint_indices(self, eef_idx: int) -> np.ndarray:
        return self._joint_indices_by_arm_id[eef_idx]

    def _get_box_pose(self) -> np.ndarray:
        extra_state = self.ppo_task_handler.get_extra_state()
        assert isinstance(extra_state, dict)
        box_pose = np.asarray(extra_state["box_pose"], dtype=np.float32).reshape(-1)
        expected_dim = self.extra_state_dims.get("box_pose")
        if expected_dim is not None and box_pose.size != expected_dim:
            raise ValueError(
                f"[{self.__class__.__name__}] box_pose dimension {box_pose.size} != {expected_dim}."
            )
        return box_pose

    def _get_box_pushpoint(self, box_pose: np.ndarray) -> np.ndarray:
        pushpoint = self.ppo_task_handler._get_box_pushpoint(box_pose)
        pushpoint = np.asarray(pushpoint, dtype=np.float32).reshape(-1)
        expected_dim = self.extra_state_dims.get("box_pushpoint")
        if expected_dim is not None and pushpoint.size != expected_dim:
            raise ValueError(
                f"[{self.__class__.__name__}] box_pushpoint dimension {pushpoint.size} != {expected_dim}."
            )
        return pushpoint

    def get_state(self):
        qpos = self.motion_manager.get_data(DataKey.MEASURED_JOINT_POS, self.obs)
        qvel = self.motion_manager.get_data(DataKey.MEASURED_JOINT_VEL, self.obs)

        qpos_ms = convert_gripper_positions_to_maniskill(
            qpos.astype(np.float32).copy(), self._gripper_joint_indices
        )
        qvel_ms = convert_gripper_velocities_to_maniskill(
            qvel.astype(np.float32).copy(), self._gripper_joint_indices
        )
        # Cache the latest joint position tensor for action denormalization
        self._latest_joint_pos_tensor = torch.as_tensor(
            qpos_ms, dtype=torch.float32, device=self.device
        )

        left_idx = self._get_arm_joint_indices(0)
        right_idx = self._get_arm_joint_indices(1)

        box_pose = self._get_box_pose()

        measured_eef_pose = self.motion_manager.get_data(
            DataKey.MEASURED_EEF_POSE, self.obs
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
            self._get_box_pushpoint(box_pose),
        ]
        state_vector = np.concatenate(
            [np.asarray(part, dtype=np.float32).reshape(-1) for part in parts]
        ).astype(np.float32)
        expected_state_dim = len(self.model_meta_info["state"]["example"])
        if state_vector.size != expected_state_dim:
            raise ValueError(
                f"[{self.__class__.__name__}] State dimension mismatch. "
                f"Constructed {state_vector.size} elements, "
                f"but model_meta_info expects {expected_state_dim}."
            )

        return torch.tensor(state_vector, dtype=torch.float32, device=self.device).unsqueeze(0)

    def infer_policy(self):
        # Infer
        if self.policy_action_buf is None or len(self.policy_action_buf) == 0:
            obs_tensor = self.get_state()

            with torch.no_grad():
                if _SAC_DETERMINISTIC:
                    raw_action = self.policy.get_eval_action(obs_tensor)
                else:
                    raw_action, _, _ = self.policy.get_action(obs_tensor)

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
                direct_joint_command = convert_gripper_tensor_to_robomanip(
                    direct_joint_command, self._gripper_joint_indices
                )

            physical_np = direct_joint_command.detach().cpu().numpy().astype(np.float64)
            if getattr(self, "_gripper_joint_indices", None) is not None and self._gripper_joint_indices.size > 0:
                physical_np[self._gripper_joint_indices] = 119.0

            self.policy_action_buf = [physical_np]

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
