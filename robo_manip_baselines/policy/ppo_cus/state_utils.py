from typing import Tuple

import numpy as np
import pinocchio as pin

def _pose_to_pos_rot6d(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float32).reshape(-1)
    if pose.size < 7:
        raise ValueError(f"EEF pose must have at least 7 elements, got {pose.size}.")
    pos = pose[0:3]
    rot = pin.Quaternion(*pose[3:7]).toRotationMatrix()
    rot6d = np.concatenate([rot[:, 0], rot[:, 1]]).astype(np.float32, copy=False)
    return np.concatenate([pos, rot6d]).astype(np.float32, copy=False)


def get_adjusted_measured_eef_pose(
    measured_eef_pose: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    eef = np.asarray(measured_eef_pose, dtype=np.float32).reshape(-1)
    left_pose = eef[0:7].copy()
    right_pose = eef[7:14].copy()

    left = (
        _pose_to_pos_rot6d(left_pose)
        if left_pose.size >= 7
        else left_pose.astype(np.float32, copy=False)
    )
    right = (
        _pose_to_pos_rot6d(right_pose)
        if right_pose.size >= 7
        else right_pose.astype(np.float32, copy=False)
    )

    return left, right
