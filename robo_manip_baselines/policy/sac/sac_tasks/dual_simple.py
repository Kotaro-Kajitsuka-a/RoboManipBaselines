"""
Helpers for the DualSimple PPO-Cus task.

This task is intentionally minimal: it does not provide any additional state
besides the joint positions / velocities that already come from the robot, so
the policy can run without marker detections or box pose estimation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional


@dataclass
class DualSimpleTask:
    rollout: "RolloutPpoCus"
    params: Mapping[str, object] = field(default_factory=dict)

    def on_reset(self) -> None:  # pragma: no cover - optional hook
        return

    def get_extra_state(self) -> Dict[str, object]:
        # No additional state is needed for this task.
        return {}


def build_ppo_task(
    rollout: "RolloutPpoCus", params: Optional[Mapping[str, object]] = None
):
    if params is None:
        params = {}
    return DualSimpleTask(rollout=rollout, params=params)
