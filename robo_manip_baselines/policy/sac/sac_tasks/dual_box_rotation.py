from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

import cv2
import numpy as np
import pyrealsense2 as rs
from cv2 import aruco

from ..RolloutSac import RolloutSac
from .dual_box_rotation_policy_state import (
    get_policy_state as get_dual_box_rotation_policy_state,
)

BOX_MARKER_ID = 2  # must match the hard-coded marker in RolloutSac
# Downward offset (marker frame -> box frame) along marker -Z axis [m]
BOX_MARKER_Z_OFFSET_M = 0.05625
# ArUco/GridBoard parameters (copied from bin/box_detection.py)
MARKER_LENGTH_M = 0.02940
MARKER_SEPARATION_M = 0.0050
MARKERS_X = 5
MARKERS_Y = 7
ARUCO_DICT_ID = aruco.DICT_4X4_50
BOX_DEPTH_M = 0.1140
GRIDBOARD_W_M = MARKERS_X * MARKER_LENGTH_M + (MARKERS_X - 1) * MARKER_SEPARATION_M
GRIDBOARD_H_M = MARKERS_Y * MARKER_LENGTH_M + (MARKERS_Y - 1) * MARKER_SEPARATION_M
BOX_HALF_SIZE = np.array([0.2170 * 0.5, 0.2845 * 0.5, 0.1140 * 0.5], dtype=np.float32)
BASE_CENTER_T_PATH = Path("robo_manip_baselines/calib/base_center_T.calib")
# Use 1280x720 (16:9) for detection stability while keeping dataset recording at 640x480.
RES_W, RES_H, FPS = 1280, 720, 30
USE_SERIAL = "314422070401"


def _rotmat_to_rpy_deg(R: np.ndarray) -> np.ndarray:
    sy = -R[2, 0]
    sy = np.clip(sy, -1.0, 1.0)
    pitch = np.arcsin(sy)
    if abs(sy) < 0.999999:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        yaw = 0.0
    return np.degrees([roll, pitch, yaw])


def _center_crop_4x3(img: np.ndarray) -> np.ndarray:
    """Center-crop a 16:9 frame to 4:3 without distortion."""
    h, w = img.shape[:2]
    target_w = int(round(h * (4.0 / 3.0)))
    if target_w >= w:
        return img
    x0 = max((w - target_w) // 2, 0)
    return img[:, x0 : x0 + target_w]


def _map_points_to_record_frame(
    points: np.ndarray, src_w: int, src_h: int, dst_w: int = 640, dst_h: int = 480
) -> np.ndarray:
    """Map points from a 16:9 source frame to the 4:3 recorded frame."""
    target_w = int(round(src_h * (4.0 / 3.0)))
    if target_w >= src_w:
        x0 = 0
        target_w = src_w
    else:
        x0 = max((src_w - target_w) // 2, 0)
    pts = points.astype(np.float32).copy()
    pts[:, 0] -= float(x0)
    scale_x = float(dst_w) / float(target_w)
    scale_y = float(dst_h) / float(src_h)
    pts[:, 0] *= scale_x
    pts[:, 1] *= scale_y
    return pts


class BoxPoseProvider:
    """RealSense + ArUco GridBoard based box pose estimator."""

    def __init__(
        self,
        serial: str = USE_SERIAL,
        calib_path: Path = BASE_CENTER_T_PATH,
        res_w: int = RES_W,
        res_h: int = RES_H,
        fps: int = FPS,
    ) -> None:
        self.serial = serial
        self.calib_path = Path(calib_path)
        self.res_w = int(res_w)
        self.res_h = int(res_h)
        self.fps = int(fps)

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._latest_transform: Optional[np.ndarray] = None
        self._latest_timestamp: Optional[float] = None
        self._latest_front_rgb: Optional[np.ndarray] = None  # RGB uint8 (480, 640, 3)
        self._latest_board_corners: Optional[np.ndarray] = None  # float32 (4, 2)
        self._latest_box_pose_seq: int = 0
        self._latest_image_seq: int = 0
        self._latest_board_seq: int = 0
        self._lock = threading.Lock()

        # Board setup
        try:
            self._aruco_dict = aruco.getPredefinedDictionary(ARUCO_DICT_ID)
        except AttributeError:
            self._aruco_dict = aruco.Dictionary_get(ARUCO_DICT_ID)
        try:
            self._aruco_params = aruco.DetectorParameters_create()
        except AttributeError:
            self._aruco_params = aruco.DetectorParameters()
        self._board = aruco.GridBoard(
            (MARKERS_X, MARKERS_Y),
            MARKER_LENGTH_M,
            MARKER_SEPARATION_M,
            self._aruco_dict,
        )

        self._base_T_cam = np.loadtxt(self.calib_path).astype(np.float32)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="box_pose_provider", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    def get_latest_box_transform(self) -> Tuple[Optional[np.ndarray], Optional[float]]:
        with self._lock:
            if self._latest_transform is None:
                return None, None
            T = self._latest_transform.copy()
            # T[2, 3] = -0.03175  # force z to fixed value at the source
            return T, self._latest_timestamp

    def get_latest_front_rgb(self) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """Return latest RGB frame resized to (640, 480) for dataset recording."""
        with self._lock:
            if self._latest_front_rgb is None:
                return None, None
            # Safe to return the stored array directly because a new array is assigned
            # on each update in the capture thread (no in-place mutation).
            return self._latest_front_rgb, self._latest_timestamp

    def get_latest_box_pose_seq(self) -> Optional[int]:
        """Return the latest box pose update sequence number."""
        with self._lock:
            return self._latest_box_pose_seq if self._latest_box_pose_seq > 0 else None

    def get_latest_image_seq(self) -> Optional[int]:
        """Return the latest image update sequence number."""
        with self._lock:
            return self._latest_image_seq if self._latest_image_seq > 0 else None

    def get_latest_board_corners(self) -> Optional[np.ndarray]:
        """Return the latest ArUco board corners in recorded-frame coordinates."""
        with self._lock:
            if self._latest_board_corners is None:
                return None
            return self._latest_board_corners.copy()

    def _run(self) -> None:
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(self.serial)
        # Request RGB frames to avoid BGR<->RGB conversion in the hot path.
        config.enable_stream(
            rs.stream.color, self.res_w, self.res_h, rs.format.rgb8, self.fps
        )

        try:
            profile = pipeline.start(config)
        except Exception as exc:  # pragma: no cover - hardware dependent
            print(
                f"[BoxPoseProvider] Failed to start RealSense pipeline: {exc}",
                flush=True,
            )
            return

        # Match normal record-data camera behavior (no manual exposure/gain override).
        # Keep rgb8 stream + 1280x720 capture path for SAC-specific cropping/resizing.

        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_stream.get_intrinsics()
        fx, fy, cx, cy = intr.fx, intr.fy, intr.ppx, intr.ppy
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        dist_coeffs = np.array(intr.coeffs[:5], dtype=np.float32)

        R_flip_x = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32)
        center_offset = np.array(
            [GRIDBOARD_W_M * 0.5, GRIDBOARD_H_M * 0.5, 0.0], dtype=np.float32
        )
        z_offset = np.array([0.0, 0.0, -BOX_DEPTH_M * 0.5], dtype=np.float32)

        try:
            while not self._stop_event.is_set():
                frames = pipeline.wait_for_frames()
                cf = frames.get_color_frame()
                if not cf:
                    continue
                img = np.asanyarray(cf.get_data())  # RGB uint8 (H, W, 3) @ RES_WxRES_H
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

                # Keep the latest RGB for recording, independent of detection success.
                # We record at 640x480 to match the typical RoboManip dataset resolution.
                rgb = _center_crop_4x3(img)
                rgb = cv2.resize(rgb, (640, 480), interpolation=cv2.INTER_AREA)

                board_corners_record = np.full((4, 2), -1.0, dtype=np.float32)

                corners, ids, _ = aruco.detectMarkers(
                    gray, self._aruco_dict, parameters=self._aruco_params
                )
                if ids is None or len(ids) == 0:
                    with self._lock:
                        self._latest_front_rgb = rgb
                        self._latest_image_seq += 1
                        self._latest_board_corners = board_corners_record
                        self._latest_board_seq += 1
                    continue
                retval, rvec, tvec = aruco.estimatePoseBoard(
                    corners, ids, self._board, K, dist_coeffs, None, None
                )
                if retval <= 0:
                    with self._lock:
                        self._latest_front_rgb = rgb
                        self._latest_image_seq += 1
                        self._latest_board_corners = board_corners_record
                        self._latest_board_seq += 1
                    continue

                board_corners_3d = np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [GRIDBOARD_W_M, 0.0, 0.0],
                        [GRIDBOARD_W_M, GRIDBOARD_H_M, 0.0],
                        [0.0, GRIDBOARD_H_M, 0.0],
                    ],
                    dtype=np.float32,
                )
                proj, _ = cv2.projectPoints(
                    board_corners_3d, rvec, tvec, K, dist_coeffs
                )
                board_corners_px = proj.reshape(-1, 2)
                board_corners_record = _map_points_to_record_frame(
                    board_corners_px, self.res_w, self.res_h
                )

                R_board, _ = cv2.Rodrigues(rvec)
                t_board_box = center_offset + R_flip_x @ z_offset
                R_cam_box = R_board @ R_flip_x
                t_cam_box = R_board @ t_board_box.reshape(3, 1) + tvec.reshape(3, 1)

                cam_T_box = np.eye(4, dtype=np.float32)
                cam_T_box[:3, :3] = R_cam_box.astype(np.float32)
                cam_T_box[:3, 3] = t_cam_box.flatten().astype(np.float32)
                base_T_box = self._base_T_cam @ cam_T_box

                with self._lock:
                    self._latest_front_rgb = rgb
                    self._latest_image_seq += 1
                    self._latest_board_corners = board_corners_record
                    self._latest_board_seq += 1
                    self._latest_transform = base_T_box.copy()
                    self._latest_timestamp = time.time()
                    self._latest_box_pose_seq += 1
        except Exception as exc:  # pragma: no cover - runtime errors
            print(f"[BoxPoseProvider] Error during detection loop: {exc}", flush=True)
        finally:
            pipeline.stop()


@dataclass
class DualBoxRotationTask:
    rollout: "RolloutSac"
    params: Mapping[str, object] = field(default_factory=dict)
    _provider: BoxPoseProvider = field(init=False, repr=False)

    def __post_init__(self) -> None:
        params = dict(self.params) if isinstance(self.params, Mapping) else {}
        provider_kwargs = {}
        for key in ("serial", "calib_path", "res_w", "res_h", "fps"):
            if key in params:
                provider_kwargs[key] = params[key]
        self._provider = BoxPoseProvider(**provider_kwargs)
        self._provider.start()
        self._pushpoint_local_right: Optional[np.ndarray] = None
        self._pushpoint_local_left: Optional[np.ndarray] = None
        self._last_box_pose: Optional[np.ndarray] = None

    def __del__(self) -> None:  # pragma: no cover - best-effort cleanup
        try:
            self._provider.stop()
        except Exception:
            pass

    def on_reset(self) -> None:
        pass

    def _rotation_matrix_to_6d(self, rotation: np.ndarray) -> np.ndarray:
        if rotation.shape != (3, 3):
            raise ValueError(
                f"[DualBoxRotationTask] Rotation matrix must be 3x3, got {rotation.shape}"
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

    def _rotation_6d_to_matrix(self, rot6d: np.ndarray) -> np.ndarray:
        rot6d = np.asarray(rot6d, dtype=np.float32).reshape(-1)
        if rot6d.size < 6:
            raise ValueError(
                f"[DualBoxRotationTask] rot6d must have 6 elements, got {rot6d.size}"
            )
        x_axis = rot6d[0:3]
        y_axis = rot6d[3:6]
        x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-8)
        y_axis = y_axis - np.dot(x_axis, y_axis) * x_axis
        y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-8)
        z_axis = np.cross(x_axis, y_axis)
        return np.stack([x_axis, y_axis, z_axis], axis=1)

    def _init_pushpoint_locals(
        self, position: np.ndarray, rotation: np.ndarray
    ) -> None:
        if self._pushpoint_local_right is not None:
            return
        hx = float(BOX_HALF_SIZE[0])
        hy = float(BOX_HALF_SIZE[1])
        long_side_offsets_local = np.array(
            [[hx, 0.0, 0.0], [-hx, 0.0, 0.0]], dtype=np.float32
        )
        long_side_offsets_world = (rotation @ long_side_offsets_local.T).T
        long_side_centers_world = position + long_side_offsets_world
        if long_side_centers_world[0, 0] >= long_side_centers_world[1, 0]:
            forward_side_center = long_side_centers_world[0]
        else:
            forward_side_center = long_side_centers_world[1]

        push_offset_local = np.array([0.0, -0.8 * hy, 0.0], dtype=np.float32)
        push_offset_world = rotation @ push_offset_local
        initial_pushpoint_by_right = forward_side_center + push_offset_world
        initial_pushpoint_by_left = 2.0 * position - initial_pushpoint_by_right

        rotation_T = rotation.T
        self._pushpoint_local_right = rotation_T @ (
            initial_pushpoint_by_right - position
        )
        self._pushpoint_local_left = rotation_T @ (initial_pushpoint_by_left - position)

    def _get_box_pushpoint(self, box_pose: np.ndarray) -> np.ndarray:
        pose = np.asarray(box_pose, dtype=np.float32).reshape(-1)
        if pose.size < 9:
            raise ValueError(
                f"[DualBoxRotationTask] box_pose must have 9 elements, got {pose.size}"
            )
        position = pose[0:3]
        rotation = self._rotation_6d_to_matrix(pose[3:9])
        self._init_pushpoint_locals(position, rotation)

        world_pushpoint_right = position + rotation @ self._pushpoint_local_right
        world_pushpoint_left = position + rotation @ self._pushpoint_local_left

        return np.concatenate([world_pushpoint_left, world_pushpoint_right]).astype(
            np.float32
        )

    def _compute_box_pose_from_marker(self) -> np.ndarray:
        T_base_to_box, ts = self._provider.get_latest_box_transform()
        if T_base_to_box is None:
            raise RuntimeError("[DualBoxRotationTask] Box pose not available yet.")
        if T_base_to_box.shape != (4, 4):
            raise ValueError(
                f"[DualBoxRotationTask] Expected 4x4 transform matrix, got shape {T_base_to_box.shape}"
            )

        translation = T_base_to_box[:3, 3].astype(np.float32)
        rotation = T_base_to_box[:3, :3]
        rotation6d = self._rotation_matrix_to_6d(rotation).astype(np.float32)
        self._last_box_pose = np.concatenate([translation, rotation6d]).astype(
            np.float32
        )
        return self._last_box_pose.copy()

    def get_extra_state(self) -> Dict[str, np.ndarray]:
        box_pose = self._compute_box_pose_from_marker()
        box_pushpoint = self._get_box_pushpoint(box_pose)
        # print(
        #     f"[DualBoxRotationTask] box_pose={box_pose}",
        #     flush=True,
        # )
        # print(
        #     f"[DualBoxRotationTask] box_pushpoint={box_pushpoint}",
        #     flush=True,
        # )
        return {"box_pose": box_pose, "box_pushpoint": box_pushpoint}

    def get_latest_front_rgb(self) -> Optional[np.ndarray]:
        """Return the latest RGB frame used by the detector thread (RGB, 640x480)."""
        rgb, _ts = self._provider.get_latest_front_rgb()
        return rgb

    def get_latest_box_pose_seq(self) -> Optional[int]:
        """Return the latest box pose update sequence number from the detector thread."""
        return self._provider.get_latest_box_pose_seq()

    def get_latest_image_seq(self) -> Optional[int]:
        """Return the latest image update sequence number from the detector thread."""
        return self._provider.get_latest_image_seq()

    def get_latest_front_aruco_board_corners(self) -> Optional[np.ndarray]:
        """Return the latest ArUco board corners in the recorded frame."""
        return self._provider.get_latest_board_corners()

    def get_record_data(self) -> Dict[str, np.ndarray | int]:
        rgb = self.get_latest_front_rgb()
        if rgb is None:
            rgb = np.zeros((480, 640, 3), dtype=np.uint8)

        board_corners = self.get_latest_front_aruco_board_corners()
        if board_corners is None:
            board_corners = np.full((4, 2), -1.0, dtype=np.float32)

        box_seq = self.get_latest_box_pose_seq()
        image_seq = self.get_latest_image_seq()
        return {
            "front_rgb_image": rgb,
            "front_aruco_board_corners": board_corners,
            "pose_estimator_seq": -1 if box_seq is None else int(box_seq),
            "image_seq": -1 if image_seq is None else int(image_seq),
        }

    def get_policy_state(self):
        return get_dual_box_rotation_policy_state(self, self.rollout)


def build_ppo_task(
    rollout: "RolloutSac", params: Optional[Mapping[str, object]] = None
):
    if params is None:
        params = {}
    return DualBoxRotationTask(rollout=rollout, params=params)


def run():
    provider = BoxPoseProvider()
    provider.start()
    print("[DualBoxRotationTask] BoxPoseProvider started. Waiting for detections...")
    frame_idx = 0
    last_t = time.time()
    try:
        while True:
            frame_idx += 1
            T_base_to_box, ts = provider.get_latest_box_transform()
            if T_base_to_box is not None:
                rpy = _rotmat_to_rpy_deg(T_base_to_box[:3, :3])
                translation_fixed = T_base_to_box[:3, 3].copy()
                ts_str = f"{ts:.3f}" if ts is not None else "n/a"
                if frame_idx % 20 == 0:
                    now = time.time()
                    fps = frame_idx / max(now - last_t, 1e-6)
                    print(
                        f"[BoxPoseProvider] fps~{fps:.1f} ts={ts_str} "
                        f"box center_fixed={translation_fixed} rpy(deg)={rpy}",
                        flush=True,
                    )
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        provider.stop()
        print("[DualBoxRotationTask] BoxPoseProvider stopped.")


if __name__ == "__main__":
    run()
