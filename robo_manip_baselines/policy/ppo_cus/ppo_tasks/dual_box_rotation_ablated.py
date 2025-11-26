from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional

import numpy as np

BOX_MARKER_ID = 2  # must match the hard-coded marker in RolloutPpoCus
# Downward offset (marker frame -> box frame) along marker -Z axis [m]
BOX_MARKER_Z_OFFSET_M = 0.05625


@dataclass
class DualBoxRotationAblatedTask:
    rollout: "RolloutPpoCus"
    params: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Marker-based box pose; no external overrides
        self.marker_id = BOX_MARKER_ID

    def _rotation_matrix_to_6d(self, rotation: np.ndarray) -> np.ndarray:
        if rotation.shape != (3, 3):
            raise ValueError(
                f"[DualBoxRotationAblatedTask] Rotation matrix must be 3x3, got {rotation.shape}"
            )
        x_axis = rotation[:, 0]
        y_axis = rotation[:, 1]

        x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-8)
        y_axis = y_axis - np.dot(x_axis, y_axis) * x_axis
        y_length = np.linalg.norm(y_axis) + 1e-8
        if y_length < 1e-6:
            # Fallback: use original y-axis if projection is degenerate
            y_axis = rotation[:, 1]
            y_length = np.linalg.norm(y_axis) + 1e-8
        y_axis = y_axis / y_length

        return np.concatenate([x_axis, y_axis])

    def _compute_box_pose_from_marker(self) -> np.ndarray:
        if self.marker_id not in self.rollout.marker_transform_cache:
            raise RuntimeError(
                f"[DualBoxRotationAblatedTask] Marker id {self.marker_id} not available in cache."
            )

        T_base_to_marker = self.rollout.marker_transform_cache[self.marker_id]
        if T_base_to_marker.shape != (4, 4):
            raise ValueError(
                f"[DualBoxRotationAblatedTask] Expected 4x4 transform matrix, got shape {T_base_to_marker.shape}"
            )

        # Rotate marker frame 180° about X, then translate box origin down along marker -Z
        T_marker_rot = np.eye(4, dtype=np.float64)
        T_marker_rot[1, 1] = -1.0
        T_marker_rot[2, 2] = -1.0

        T_marker_to_box = np.eye(4, dtype=np.float64)
        T_marker_to_box[2, 3] = -float(BOX_MARKER_Z_OFFSET_M)

        T_base_to_box = T_base_to_marker @ T_marker_rot @ T_marker_to_box

        translation = T_base_to_box[:3, 3].astype(np.float32)
        translation[1] += 0.3291  # emergency offset
        rotation = T_base_to_box[:3, :3]
        rotation6d = self._rotation_matrix_to_6d(rotation).astype(np.float32)
        return np.concatenate([translation, rotation6d]).astype(np.float32)

    def get_extra_state(self) -> Dict[str, np.ndarray]:
        box_pose = self._compute_box_pose_from_marker()
        print(
            f"[DualBoxRotationAblatedTask] box_pose={box_pose}",
            flush=True,
        )
        return {"box_pose": box_pose}


def build_ppo_task(
    rollout: "RolloutPpoCus", params: Optional[Mapping[str, object]] = None
):
    if params is None:
        params = {}
    return DualBoxRotationAblatedTask(rollout=rollout, params=params)
