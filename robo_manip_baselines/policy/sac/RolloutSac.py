import csv
import importlib
import os
from typing import Any, Dict

import cv2
import matplotlib.pylab as plt
import numpy as np
import torch

from robo_manip_baselines.common import (
    DataKey,
    RolloutBase,
)

from . import PpoUtil, SacUtil
from .gripper_utils import (
    convert_gripper_positions_to_maniskill,
)

_NORMALIZED_ACTION_LOW = torch.tensor(-1.0, dtype=torch.float32)
_NORMALIZED_ACTION_HIGH = torch.tensor(1.0, dtype=torch.float32)
_DEFAULT_ARM_JOINT_DELTA_LIMIT = 0.045
_DEFAULT_GRIPPER_JOINT_DELTA_LIMIT = 0.1
_FIXED_GRIPPER_COMMAND = 119.0
_DETERMINISTIC = True
_USE_CUDA = torch.cuda.is_available()
_ROLLOUT_TASK_MODULES = {
    "DualBoxRotation": "robo_manip_baselines.policy.sac.sac_tasks.dual_box_rotation",
    "TrashBinRolling": "robo_manip_baselines.policy.sac.sac_tasks.single_aruco_marker",
    "DualSimple": "robo_manip_baselines.policy.sac.sac_tasks.dual_simple",
    "Align": "robo_manip_baselines.policy.sac.sac_tasks.align",
    "JointHoldMarkerCheck": "robo_manip_baselines.policy.sac.sac_tasks.jointhold_marker_check",
}
_ROLLOUT_TASK_NAME_ALIASES = {
    "dual_box_rotation": "DualBoxRotation",
    "single_aruco_marker": "TrashBinRolling",
    "trash_bin_rolling": "TrashBinRolling",
    "dual_simple": "DualSimple",
    "align": "Align",
    "jointhold_marker_check": "JointHoldMarkerCheck",
}


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
        self.policy_algo = str(
            self.model_meta_info.get("policy", {}).get("algo", "sac")
        ).lower()
        self._init_joint_metadata()
        self.extra_state_keys: list[str] = []
        self.extra_state_dims: Dict[str, int] = {}
        self.ppo_task_handler = None
        self.ppo_task_params: Dict[str, Any] = {}

        # Backward-compat attribute for tasks that may expect it; left empty.
        self.marker_transform_cache: Dict[int, np.ndarray] = {}
        self._setup_task_from_meta()

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

    def _setup_task_from_meta(self) -> None:
        task_cfg = self.model_meta_info.get("ppo_task") or self.model_meta_info.get(
            "rl_task"
        )
        if not task_cfg and self._is_dual_marker_policy():
            task_cfg = {
                "name": "TrashBinRolling",
                "params": {},
            }
        if not task_cfg:
            return

        extra_keys_cfg = task_cfg.get("extra_keys", [])
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

        params = task_cfg.get("params") or {}
        assert isinstance(params, dict)
        self.ppo_task_params = dict(params)

        task_name = self._get_rollout_task_name(task_cfg)
        module_path = _ROLLOUT_TASK_MODULES[task_name]

        if self.args.yolo_pt:
            if task_name != "DualBoxRotation":
                raise ValueError(
                    f"[{self.__class__.__name__}] --yolo_pt is only supported for "
                    f"DualBoxRotation, got {task_name}."
                )
            module_path = "robo_manip_baselines.policy.sac.sac_tasks.box_estimator"
            self.ppo_task_params["pt_path"] = self.args.yolo_pt
            self.ppo_task_params["camera_name"] = "top"

        try:
            task_module = importlib.import_module(module_path)
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                f"[{self.__class__.__name__}] Failed to import PPO task module '{module_path}'."
            ) from exc

        self.extra_state_keys = extra_state_keys
        self.extra_state_dims = extra_state_dims
        self.ppo_task_handler = task_module.build_ppo_task(self, self.ppo_task_params)
        assert self.ppo_task_handler is not None

        print(
            f"[{self.__class__.__name__}] Loaded rollout task '{task_name}'. "
            f"Extra state keys: {self.extra_state_keys}"
        )

    def _get_rollout_task_name(self, task_cfg: Dict[str, Any]) -> str:
        raw_name = task_cfg.get("name")
        if raw_name is None:
            module_path = task_cfg.get("module")
            assert isinstance(module_path, str) and module_path
            raw_name = module_path.rsplit(".", 1)[-1]

        assert isinstance(raw_name, str) and raw_name
        task_name = _ROLLOUT_TASK_NAME_ALIASES.get(raw_name, raw_name)
        if task_name not in _ROLLOUT_TASK_MODULES:
            known = ", ".join(sorted(_ROLLOUT_TASK_MODULES))
            raise ValueError(
                f"[{self.__class__.__name__}] Unsupported rollout task name "
                f"'{raw_name}'. Known tasks: {known}."
            )
        return task_name

    def _is_dual_marker_policy(self) -> bool:
        return list(self.state_keys) == [
            "left_measured_joint_pos",
            "left_measured_joint_vel",
            "right_measured_joint_pos",
            "right_measured_joint_vel",
            "marker_position",
            "marker_rotation_6d",
        ]

    def _init_action_scaling_tensors(self) -> None:
        """Initialize tensors describing per-joint min/max and delta limits."""
        low_np = np.asarray(self.env.action_space.low, dtype=np.float32).reshape(-1)
        high_np = np.asarray(self.env.action_space.high, dtype=np.float32).reshape(-1)
        if low_np.size != self.action_dim or high_np.size != self.action_dim:
            raise ValueError(
                f"[{self.__class__.__name__}] action space dimension mismatch: "
                f"low={low_np.size}, high={high_np.size}, meta={self.action_dim}."
            )
        low_np = convert_gripper_positions_to_maniskill(
            low_np.copy(), self._gripper_joint_indices
        )
        high_np = convert_gripper_positions_to_maniskill(
            high_np.copy(), self._gripper_joint_indices
        )
        self._joint_position_low = torch.as_tensor(
            np.minimum(low_np, high_np), dtype=torch.float32, device=self.device
        )
        self._joint_position_high = torch.as_tensor(
            np.maximum(low_np, high_np), dtype=torch.float32, device=self.device
        )

        delta_limit = torch.full(
            (self.action_dim,),
            _DEFAULT_ARM_JOINT_DELTA_LIMIT,
            dtype=torch.float32,
            device=self.device,
        )
        if self._gripper_joint_indices.size > 0:
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
        state_dict = (
            raw_ckpt["actor"]
            if isinstance(raw_ckpt, dict) and "actor" in raw_ckpt
            else raw_ckpt
        )

        if self.policy_algo == "sac":
            self.policy = SacUtil.build_policy(self.model_meta_info)
        elif self.policy_algo == "ppo":
            self.policy = PpoUtil.build_policy(
                state_dict, self.state_dim, self.action_dim
            )
        else:
            raise ValueError(
                f"[{self.__class__.__name__}] Unsupported policy algo: {self.policy_algo}"
            )
        self.policy.load_state_dict(state_dict)

        use_cuda = _USE_CUDA and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.policy.to(self.device)
        self.policy.eval()
        self._deterministic = _DETERMINISTIC
        self._fixed_gripper_command = _FIXED_GRIPPER_COMMAND

        self._normalized_action_low = torch.full(
            (self.action_dim,), float(_NORMALIZED_ACTION_LOW.item()), device=self.device
        )
        self._normalized_action_high = torch.full(
            (self.action_dim,),
            float(_NORMALIZED_ACTION_HIGH.item()),
            device=self.device,
        )
        self._init_action_scaling_tensors()

        print(
            f"[{self.__class__.__name__}] Load ManiSkill {self.policy_algo.upper()} checkpoint on {self.device}"
        )
        self._setup_state_action_csv()

    def _setup_state_action_csv(self):
        checkpoint_dir = os.path.dirname(os.path.abspath(self.args.checkpoint))
        self._state_action_csv_path = os.path.join(
            checkpoint_dir, "rollout_sac_state_action.csv"
        )
        with open(self._state_action_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["episode_idx", "rollout_step", "time"]
                + [f"state_{idx}" for idx in range(self.state_dim)]
                + [f"action_command_joint_pos_{idx}" for idx in range(self.action_dim)]
            )
        print(
            f"[{self.__class__.__name__}] Save state/action CSV: {self._state_action_csv_path}"
        )

    def append_state_action_csv(self, state_tensor, action):
        state = state_tensor.squeeze(0).detach().cpu().numpy().astype(float).reshape(-1)
        action = np.asarray(action, dtype=float).reshape(-1)
        if state.size != self.state_dim:
            raise ValueError(
                f"[{self.__class__.__name__}] state CSV dim mismatch: "
                f"{state.size} != {self.state_dim}."
            )
        if action.size != self.action_dim:
            raise ValueError(
                f"[{self.__class__.__name__}] action CSV dim mismatch: "
                f"{action.size} != {self.action_dim}."
            )

        episode_idx = int(getattr(self.data_manager, "episode_idx", 0))
        rollout_step = int(getattr(self, "rollout_time_idx", 0))
        elapsed_time = self.phase_manager.phase.get_elapsed_duration()
        with open(self._state_action_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [episode_idx, rollout_step, float(elapsed_time)]
                + state.tolist()
                + action.tolist()
            )

    def setup_variables(self):
        super().setup_variables()

        # Disable vision/image pipelines for this rollout; box pose is handled in the task.
        self.camera_names = []
        self._detector_camera_name = "top" if self.args.yolo_pt else "front"
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
        self._pose_estimator_seq_prev = None
        self._pose_estimator_stagnation = 0
        self._image_seq_prev = None
        self._image_stagnation = 0

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
        if stagnation >= 16:
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

        record = self.ppo_task_handler.get_record_data()
        for key, value in record.items():
            self.data_manager.append_single_data(key, value)

        pose_estimator_seq = int(record["pose_estimator_seq"])
        image_seq = int(record["image_seq"])
        self._pose_estimator_seq_prev, self._pose_estimator_stagnation = (
            self._update_stagnation(
                "pose estimator",
                None if pose_estimator_seq < 0 else pose_estimator_seq,
                self._pose_estimator_seq_prev,
                self._pose_estimator_stagnation,
            )
        )
        self._image_seq_prev, self._image_stagnation = self._update_stagnation(
            "image capture",
            None if image_seq < 0 else image_seq,
            self._image_seq_prev,
            self._image_stagnation,
        )

    def _get_arm_joint_indices(self, eef_idx: int) -> np.ndarray:
        return self._joint_indices_by_arm_id[eef_idx]

    def get_state(self):
        return self.ppo_task_handler.get_policy_state()

    def infer_policy(self):
        if self.policy_algo == "ppo":
            PpoUtil.infer_policy(self)
            return
        SacUtil.infer_policy(self)

    def reset(self):
        super().reset()
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
