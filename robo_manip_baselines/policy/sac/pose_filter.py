from typing import Tuple

import numpy as np

DEFAULT_MAX_POSITION_JUMP_M = 0.012


class PoseJumpRejector:
    def __init__(self, max_position_jump_m: float = DEFAULT_MAX_POSITION_JUMP_M):
        self.max_position_jump_m = float(max_position_jump_m)
        if self.max_position_jump_m <= 0.0:
            raise ValueError(
                f"max_position_jump_m must be positive, got {self.max_position_jump_m}."
            )
        self._last_pose = None

    def reset(self) -> None:
        self._last_pose = None

    def update(self, pose: np.ndarray) -> Tuple[np.ndarray, bool]:
        pose = np.asarray(pose, dtype=np.float32).reshape(4, 4)
        if self._last_pose is None:
            self._last_pose = pose.copy()
            return pose, True

        position_jump = np.linalg.norm(pose[:3, 3] - self._last_pose[:3, 3])
        if position_jump > self.max_position_jump_m:
            return self._last_pose.copy(), False

        self._last_pose = pose.copy()
        return pose, True
