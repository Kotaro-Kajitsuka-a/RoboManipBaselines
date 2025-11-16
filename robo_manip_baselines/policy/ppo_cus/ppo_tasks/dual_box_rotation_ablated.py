from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional

import numpy as np

DEFAULT_BOX_POSE = np.array(
    [0.44, -0.01, 0.04, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32
)


@dataclass
class DualBoxRotationAblatedTask:
    rollout: "RolloutPpoCus"
    params: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        pose_override = self.params.get("box_pose_const")
        if pose_override is None:
            self._box_pose = DEFAULT_BOX_POSE.copy()
        else:
            pose_arr = np.asarray(pose_override, dtype=np.float32).reshape(-1)
            if pose_arr.size != DEFAULT_BOX_POSE.size:
                raise ValueError(
                    f"[DualBoxRotationAblatedTask] Expected {DEFAULT_BOX_POSE.size} values for box pose, "
                    f"got {pose_arr.size}"
                )
            self._box_pose = pose_arr.copy()



    def get_extra_state(self) -> Dict[str, np.ndarray]:
        return {"box_pose": self._box_pose.copy()}


def build_ppo_task(
    rollout: "RolloutPpoCus", params: Optional[Mapping[str, object]] = None
):
    if params is None:
        params = {}
    return DualBoxRotationAblatedTask(rollout=rollout, params=params)
