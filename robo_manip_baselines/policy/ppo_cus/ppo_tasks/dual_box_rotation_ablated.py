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

BOX_MARKER_ID = 2  # must match the hard-coded marker in RolloutPpoCus
# Downward offset (marker frame -> box frame) along marker -Z axis [m]
BOX_MARKER_Z_OFFSET_M = 0.05625
# ArUco/GridBoard parameters (copied from bin/box_detection.py)
BIG_MARKER_LENGTH_M = 0.02940
BIG_MARKER_SEPARATION_M = 0.0050
BIG_MARKERS_X = 5
BIG_MARKERS_Y = 7
BIG_ARUCO_DICT_ID = aruco.DICT_4X4_50
SMALL_BOARD_DICT_ID = aruco.DICT_5X5_250
SMALL_BOARD_MARKER_IDS = [100, 101, 102]
SMALL_BOARD_MARKER_LENGTH_M = 0.0287
SMALL_BOARD_MARKER_GAP_M = 0.005
SMALL_BOARD_W_M = (
    SMALL_BOARD_MARKER_LENGTH_M * len(SMALL_BOARD_MARKER_IDS)
    + SMALL_BOARD_MARKER_GAP_M * (len(SMALL_BOARD_MARKER_IDS) - 1)
)
SMALL_BOARD_H_M = SMALL_BOARD_MARKER_LENGTH_M
SMALL_BOARD_TARGET_Y_OFFSET_M = -0.0825
SMALL_BOARD_TARGET_RY_OFFSET_RAD = np.deg2rad(-90.0)
BIG_PANEL_Z_OFFSET_M = 0.003
BIG_BOARD_RZ_OFFSET_RAD = np.deg2rad(90.0)
BASE_CENTER_T_PATH = Path("robo_manip_baselines/calib/base_center_T.calib")
RES_W, RES_H, FPS = 1920, 1080, 30
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


def _rotation_z(rad: float) -> np.ndarray:
    c = np.cos(rad)
    s = np.sin(rad)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def _rotation_y(rad: float) -> np.ndarray:
    c = np.cos(rad)
    s = np.sin(rad)
    return np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=np.float32,
    )


def _get_aruco_dictionary(dict_id):
    try:
        return aruco.getPredefinedDictionary(dict_id)
    except AttributeError:
        return aruco.Dictionary_get(dict_id)


def _get_aruco_parameters():
    try:
        return aruco.DetectorParameters_create()
    except AttributeError:
        return aruco.DetectorParameters()


def _detect_markers(gray, aruco_dict, aruco_params):
    if hasattr(aruco, "ArucoDetector"):
        detector = aruco.ArucoDetector(aruco_dict, aruco_params)
        return detector.detectMarkers(gray)
    return aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)


def _build_small_board(small_board_dict):
    return aruco.GridBoard(
        (len(SMALL_BOARD_MARKER_IDS), 1),
        SMALL_BOARD_MARKER_LENGTH_M,
        SMALL_BOARD_MARKER_GAP_M,
        small_board_dict,
        np.array(SMALL_BOARD_MARKER_IDS, dtype=np.int32),
    )


def _estimate_small_board_pose(corners, ids, board, K, dist_coeffs):
    if ids is None or len(ids) == 0:
        return None, None

    retval, rvec, tvec = aruco.estimatePoseBoard(
        corners, ids, board, K, dist_coeffs, None, None
    )
    if retval <= 0:
        return None, None

    R_board, _ = cv2.Rodrigues(rvec)
    center_offset = np.array(
        [SMALL_BOARD_W_M * 0.5, SMALL_BOARD_H_M * 0.5, 0.0], dtype=np.float32
    )
    t_center = R_board @ center_offset.reshape(3, 1) + tvec.reshape(3, 1)
    target_offset = np.array(
        [0.0, SMALL_BOARD_TARGET_Y_OFFSET_M, 0.0], dtype=np.float32
    )
    R_target = R_board @ _rotation_y(SMALL_BOARD_TARGET_RY_OFFSET_RAD)
    t_target = R_board @ target_offset.reshape(3, 1) + t_center.reshape(3, 1)
    rvec_target, _ = cv2.Rodrigues(R_target)
    return rvec_target.reshape(3, 1), t_target.reshape(3, 1)


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
        self._latest_small_board_transform: Optional[np.ndarray] = None
        self._latest_timestamp: Optional[float] = None
        self._latest_small_board_timestamp: Optional[float] = None
        self._lock = threading.Lock()

        # Board setup
        self._aruco_dict = _get_aruco_dictionary(BIG_ARUCO_DICT_ID)
        self._small_board_dict = _get_aruco_dictionary(SMALL_BOARD_DICT_ID)
        self._aruco_params = _get_aruco_parameters()
        self._small_board = _build_small_board(self._small_board_dict)
        self._board = aruco.GridBoard(
            (BIG_MARKERS_X, BIG_MARKERS_Y), BIG_MARKER_LENGTH_M, BIG_MARKER_SEPARATION_M, self._aruco_dict
        )
        self._board_w = BIG_MARKERS_X * BIG_MARKER_LENGTH_M + (BIG_MARKERS_X - 1) * BIG_MARKER_SEPARATION_M
        self._board_h = BIG_MARKERS_Y * BIG_MARKER_LENGTH_M + (BIG_MARKERS_Y - 1) * BIG_MARKER_SEPARATION_M

        self._base_T_cam = np.loadtxt(self.calib_path).astype(np.float32)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="box_pose_provider", daemon=True)
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
            #T[2, 3] = -0.03175  # force z to fixed value at the source
            return T, self._latest_timestamp

    def get_latest_small_board_transform(self) -> Tuple[Optional[np.ndarray], Optional[float]]:
        with self._lock:
            if self._latest_small_board_transform is None:
                return None, None
            return self._latest_small_board_transform.copy(), self._latest_small_board_timestamp

    def _run(self) -> None:
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(self.serial)
        config.enable_stream(rs.stream.color, self.res_w, self.res_h, rs.format.bgr8, self.fps)

        try:
            profile = pipeline.start(config)
        except Exception as exc:  # pragma: no cover - hardware dependent
            print(f"[BoxPoseProvider] Failed to start RealSense pipeline: {exc}", flush=True)
            return

        # Match normal record-data behavior: keep camera-side default exposure/gain control.

        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_stream.get_intrinsics()
        fx, fy, cx, cy = intr.fx, intr.fy, intr.ppx, intr.ppy
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        dist_coeffs = np.array(intr.coeffs[:5], dtype=np.float32)

        R_flip_x = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32)
        R_box_z_offset = _rotation_z(BIG_BOARD_RZ_OFFSET_RAD)
        center_offset = np.array([self._board_w * 0.5, self._board_h * 0.5, 0.0], dtype=np.float32)
        z_offset = np.array([0.0, 0.0, -BIG_PANEL_Z_OFFSET_M], dtype=np.float32)

        try:
            while not self._stop_event.is_set():
                frames = pipeline.wait_for_frames()
                cf = frames.get_color_frame()
                if not cf:
                    continue
                img = np.asanyarray(cf.get_data())
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                small_corners, small_ids, _ = _detect_markers(
                    gray, self._small_board_dict, self._aruco_params
                )
                small_rvec, small_tvec = _estimate_small_board_pose(
                    small_corners,
                    small_ids,
                    self._small_board,
                    K,
                    dist_coeffs,
                )
                if small_rvec is not None and small_tvec is not None:
                    R_small, _ = cv2.Rodrigues(small_rvec)
                    cam_T_small = np.eye(4, dtype=np.float32)
                    cam_T_small[:3, :3] = R_small.astype(np.float32)
                    cam_T_small[:3, 3] = small_tvec.flatten().astype(np.float32)
                    base_T_small = self._base_T_cam @ cam_T_small
                    with self._lock:
                        self._latest_small_board_transform = base_T_small.copy()
                        self._latest_small_board_timestamp = time.time()

                corners, ids, _ = _detect_markers(gray, self._aruco_dict, self._aruco_params)

                if ids is None or len(ids) == 0:
                    continue
                retval, rvec, tvec = aruco.estimatePoseBoard(corners, ids, self._board, K, dist_coeffs, None, None)
                if retval <= 0:
                    continue

                R_board, _ = cv2.Rodrigues(rvec)
                t_board_box = center_offset + R_flip_x @ z_offset
                R_cam_box = R_board @ R_flip_x @ R_box_z_offset
                t_cam_box = R_board @ t_board_box.reshape(3, 1) + tvec.reshape(3, 1)

                cam_T_box = np.eye(4, dtype=np.float32)
                cam_T_box[:3, :3] = R_cam_box.astype(np.float32)
                cam_T_box[:3, 3] = t_cam_box.flatten().astype(np.float32)
                base_T_box = self._base_T_cam @ cam_T_box

                with self._lock:
                    self._latest_transform = base_T_box.copy()
                    self._latest_timestamp = time.time()
        except Exception as exc:  # pragma: no cover - runtime errors
            print(f"[BoxPoseProvider] Error during detection loop: {exc}", flush=True)
        finally:
            pipeline.stop()


@dataclass
class DualBoxRotationAblatedTask:
    rollout: "RolloutPpoCus"
    params: Mapping[str, object] = field(default_factory=dict)
    _provider: BoxPoseProvider = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._provider = BoxPoseProvider()
        self._provider.start()

    def __del__(self) -> None:  # pragma: no cover - best-effort cleanup
        try:
            self._provider.stop()
        except Exception:
            pass

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
        T_base_to_box, ts = self._provider.get_latest_box_transform()
        if T_base_to_box is None:
            raise RuntimeError("[DualBoxRotationAblatedTask] Box pose not available yet.")
        if T_base_to_box.shape != (4, 4):
            raise ValueError(
                f"[DualBoxRotationAblatedTask] Expected 4x4 transform matrix, got shape {T_base_to_box.shape}"
            )

        translation = T_base_to_box[:3, 3].astype(np.float32)
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


def run():
    provider = BoxPoseProvider()
    provider.start()
    print("[DualBoxRotationAblatedTask] BoxPoseProvider started. Waiting for detections...")
    frame_idx = 0
    last_t = time.time()
    try:
        while True:
            frame_idx += 1
            T_base_to_box, ts = provider.get_latest_box_transform()
            T_base_to_small, small_ts = provider.get_latest_small_board_transform()
            if T_base_to_box is not None:
                rpy = _rotmat_to_rpy_deg(T_base_to_box[:3, :3])
                translation_fixed = T_base_to_box[:3, 3].copy()
                ts_str = f"{ts:.3f}" if ts is not None else "n/a"
                if frame_idx % 20 == 0:
                    now = time.time()
                    fps = frame_idx / max(now - last_t, 1e-6)
                    print(
                        f"[BoxPoseProvider] fps~{fps:.1f} ts={ts_str} box center_fixed={translation_fixed} rpy(deg)={rpy}",
                        flush=True,
                    )
            if T_base_to_small is not None and frame_idx % 20 == 0:
                small_rpy = _rotmat_to_rpy_deg(T_base_to_small[:3, :3])
                small_translation = T_base_to_small[:3, 3].copy()
                small_ts_str = f"{small_ts:.3f}" if small_ts is not None else "n/a"
                print(
                    f"[SmallBoard] ids={SMALL_BOARD_MARKER_IDS} ts={small_ts_str} "
                    f"center={small_translation} rpy(deg)={small_rpy}",
                    flush=True,
                )
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        provider.stop()
        print("[DualBoxRotationAblatedTask] BoxPoseProvider stopped.")




if __name__ == "__main__":
    run()
