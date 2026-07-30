import numpy as np
from scipy.spatial.transform import Rotation


def _normalize_vector(vec):
    norm = np.linalg.norm(vec, axis=-1, keepdims=True)
    return vec / np.clip(norm, 1e-8, None)


def _get_rotation_6d_from_matrix(rot):
    return rot[..., :2, :].reshape(*rot.shape[:-2], 6)


def _get_matrix_from_rotation_6d(rot6):
    a1 = rot6[..., 0:3]
    a2 = rot6[..., 3:6]

    b1 = _normalize_vector(a1)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = _normalize_vector(b2)
    b3 = np.cross(b1, b2)

    return np.stack([b1, b2, b3], axis=-2)


def get_pose9_from_pose7(pose):
    """Get pose (tx, ty, tz, r6...) from pose (tx, ty, tz, qw, qx, qy, qz)."""
    pose = np.asarray(pose)
    if pose.shape[-1] % 7 != 0:
        raise ValueError(
            f"[get_pose9_from_pose7] Last dimension must be divisible by 7: {pose.shape}"
        )

    num_pose = pose.shape[-1] // 7
    pose7 = pose.reshape(*pose.shape[:-1], num_pose, 7)
    pos = pose7[..., :3]
    quat = pose7[..., 3:7]

    rot = Rotation.from_quat(quat[..., [1, 2, 3, 0]].reshape(-1, 4)).as_matrix()
    rot = rot.reshape(*pose7.shape[:-1], 3, 3)
    rot6 = _get_rotation_6d_from_matrix(rot)
    pose9 = np.concatenate([pos, rot6], axis=-1)

    return pose9.reshape(*pose.shape[:-1], 9 * num_pose)


def get_pose7_from_pose9(pose):
    """Get pose (tx, ty, tz, qw, qx, qy, qz) from pose (tx, ty, tz, r6...)."""
    pose = np.asarray(pose)
    if pose.shape[-1] % 9 != 0:
        raise ValueError(
            f"[get_pose7_from_pose9] Last dimension must be divisible by 9: {pose.shape}"
        )

    num_pose = pose.shape[-1] // 9
    pose9 = pose.reshape(*pose.shape[:-1], num_pose, 9)
    pos = pose9[..., :3]
    rot6 = pose9[..., 3:9]

    rot = _get_matrix_from_rotation_6d(rot6)
    quat = Rotation.from_matrix(rot.reshape(-1, 3, 3)).as_quat()
    quat = quat.reshape(*pose9.shape[:-1], 4)[..., [3, 0, 1, 2]]
    quat = np.where(quat[..., :1] < 0.0, -quat, quat)
    pose7 = np.concatenate([pos, quat], axis=-1)

    return pose7.reshape(*pose.shape[:-1], 7 * num_pose)
