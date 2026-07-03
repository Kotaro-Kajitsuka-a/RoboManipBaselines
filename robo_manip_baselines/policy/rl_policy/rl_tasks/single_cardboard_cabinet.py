from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Mapping, Optional, Tuple

import cv2
import numpy as np
from cv2 import aruco

from robo_manip_baselines.policy.rl_policy.rl_tasks.cabinet_marker_detection import (
    BASE_CENTER_T_PATH,
    BIG_ARUCO_DICT_ID,
    FPS,
    RES_H,
    RES_W,
    SMALL_BOARD_DICT_ID,
    SMALL_BOARD_MARKER_IDS,
    USE_SERIAL,
    build_big_board,
    build_small_board,
    estimate_big_board_outer_pose,
    estimate_small_board_pose,
    transform_from_rvec_tvec,
)

if TYPE_CHECKING:
    from robo_manip_baselines.policy.rl_policy.RolloutRLPolicy import RolloutRLPolicy


def _center_crop_4x3(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    target_w = int(round(h * (4.0 / 3.0)))
    if target_w >= w:
        return img
    x0 = max((w - target_w) // 2, 0)
    return img[:, x0 : x0 + target_w]


def _rotation_matrix_to_6d(rotation: np.ndarray) -> np.ndarray:
    x_axis = rotation[:, 0]
    y_axis = rotation[:, 1]
    x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-8)
    y_axis = y_axis - np.dot(x_axis, y_axis) * x_axis
    y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-8)
    return np.concatenate([x_axis, y_axis]).astype(np.float32)


def _rotmat_to_rpy_deg(rotation: np.ndarray) -> np.ndarray:
    sy = -rotation[2, 0]
    sy = np.clip(sy, -1.0, 1.0)
    pitch = np.arcsin(sy)
    if abs(sy) < 0.999999:
        roll = np.arctan2(rotation[2, 1], rotation[2, 2])
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = np.arctan2(-rotation[1, 2], rotation[1, 1])
        yaw = 0.0
    return np.degrees([roll, pitch, yaw])


class CabinetMarkerPoseProvider:
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
        self._latest_outer_transform: Optional[np.ndarray] = None
        self._latest_inner_transform: Optional[np.ndarray] = None
        self._latest_front_rgb: Optional[np.ndarray] = None
        self._latest_outer_marker_seq = 0
        self._latest_inner_marker_seq = 0
        self._latest_image_seq = 0
        self._lock = threading.Lock()

        self._big_dict = aruco.getPredefinedDictionary(BIG_ARUCO_DICT_ID)
        self._small_dict = aruco.getPredefinedDictionary(SMALL_BOARD_DICT_ID)
        self._aruco_params = aruco.DetectorParameters()
        self._big_detector = aruco.ArucoDetector(self._big_dict, self._aruco_params)
        self._small_detector = aruco.ArucoDetector(
            self._small_dict, self._aruco_params
        )
        self._small_board = build_small_board(self._small_dict)
        self._big_board = build_big_board(self._big_dict)
        self._base_T_cam = np.loadtxt(self.calib_path).astype(np.float32)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="cabinet_marker_pose_provider", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    def get_latest_outer_transform(self) -> Tuple[Optional[np.ndarray], Optional[int]]:
        with self._lock:
            if self._latest_outer_transform is None:
                return None, None
            return self._latest_outer_transform.copy(), self._latest_outer_marker_seq

    def get_latest_inner_transform(self) -> Tuple[Optional[np.ndarray], Optional[int]]:
        with self._lock:
            if self._latest_inner_transform is None:
                return None, None
            return self._latest_inner_transform.copy(), self._latest_inner_marker_seq

    def get_latest_front_rgb(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._latest_front_rgb is None:
                return None
            return self._latest_front_rgb.copy()

    def get_latest_outer_marker_seq(self) -> Optional[int]:
        with self._lock:
            return (
                self._latest_outer_marker_seq
                if self._latest_outer_marker_seq > 0
                else None
            )

    def get_latest_inner_marker_seq(self) -> Optional[int]:
        with self._lock:
            return (
                self._latest_inner_marker_seq
                if self._latest_inner_marker_seq > 0
                else None
            )

    def get_latest_image_seq(self) -> Optional[int]:
        with self._lock:
            return self._latest_image_seq if self._latest_image_seq > 0 else None

    def _run(self) -> None:
        import pyrealsense2 as rs

        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(self.serial)
        config.enable_stream(
            rs.stream.color, self.res_w, self.res_h, rs.format.bgr8, self.fps
        )

        try:
            profile = pipeline.start(config)
        except Exception as exc:
            print(
                f"[CabinetMarkerPoseProvider] Failed to start RealSense pipeline: {exc}",
                flush=True,
            )
            return

        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_stream.get_intrinsics()
        K = np.array(
            [[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]],
            dtype=np.float32,
        )
        dist_coeffs = np.array(intr.coeffs[:5], dtype=np.float32)

        try:
            while not self._stop_event.is_set():
                frames = pipeline.wait_for_frames()
                cf = frames.get_color_frame()
                if not cf:
                    continue
                img = np.asanyarray(cf.get_data())
                rgb = cv2.cvtColor(_center_crop_4x3(img), cv2.COLOR_BGR2RGB)
                rgb = cv2.resize(rgb, (640, 480), interpolation=cv2.INTER_AREA)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                with self._lock:
                    self._latest_front_rgb = rgb
                    self._latest_image_seq += 1

                small_corners, small_ids, _ = self._small_detector.detectMarkers(gray)
                small_rvec, small_tvec = estimate_small_board_pose(
                    small_corners,
                    small_ids,
                    self._small_board,
                    K,
                    dist_coeffs,
                )
                if small_rvec is not None and small_tvec is not None:
                    cam_T_inner = transform_from_rvec_tvec(small_rvec, small_tvec)
                    base_T_inner = self._base_T_cam @ cam_T_inner
                    with self._lock:
                        self._latest_inner_transform = base_T_inner.copy()
                        self._latest_inner_marker_seq += 1

                corners, ids, _ = self._big_detector.detectMarkers(gray)
                outer_rvec, outer_tvec = estimate_big_board_outer_pose(
                    corners,
                    ids,
                    self._big_board,
                    K,
                    dist_coeffs,
                )
                if outer_rvec is None or outer_tvec is None:
                    continue

                cam_T_outer = transform_from_rvec_tvec(outer_rvec, outer_tvec)
                base_T_outer = self._base_T_cam @ cam_T_outer
                with self._lock:
                    self._latest_outer_transform = base_T_outer.copy()
                    self._latest_outer_marker_seq += 1
        except Exception as exc:
            print(
                f"[CabinetMarkerPoseProvider] Error during detection loop: {exc}",
                flush=True,
            )
        finally:
            pipeline.stop()


@dataclass
class SingleCardboardCabinetTask:
    rollout: "RolloutRLPolicy"
    params: Mapping[str, object] = field(default_factory=dict)
    _provider: CabinetMarkerPoseProvider = field(init=False, repr=False)

    def __post_init__(self) -> None:
        params = dict(self.params) if isinstance(self.params, Mapping) else {}
        provider_kwargs = {}
        for key in ("serial", "calib_path", "res_w", "res_h", "fps"):
            if key in params:
                provider_kwargs[key] = params[key]
        self._provider = CabinetMarkerPoseProvider(**provider_kwargs)
        self._provider.start()

    def __del__(self) -> None:
        try:
            self._provider.stop()
        except Exception:
            pass

    def _panel_state(
        self, transform: np.ndarray, name: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        if transform.shape != (4, 4):
            raise ValueError(f"{name} transform must be 4x4, got {transform.shape}.")
        position = transform[:3, 3].astype(np.float32)
        rotation_6d = _rotation_matrix_to_6d(transform[:3, :3])
        return position, rotation_6d

    def get_extra_state(self) -> Dict[str, np.ndarray]:
        outer_T, _outer_seq = self._provider.get_latest_outer_transform()
        inner_T, _inner_seq = self._provider.get_latest_inner_transform()
        if outer_T is None:
            raise RuntimeError(
                "[SingleCardboardCabinetTask] Outer marker panel pose not available yet."
            )
        if inner_T is None:
            raise RuntimeError(
                "[SingleCardboardCabinetTask] Inner marker panel pose not available yet."
            )

        inner_pos, inner_rot6d = self._panel_state(inner_T, "inner marker panel")
        outer_pos, outer_rot6d = self._panel_state(outer_T, "outer marker panel")
        return {
            "extra/inner_marker_panel_position": inner_pos,
            "extra/inner_marker_panel_rotation_6d": inner_rot6d,
            "extra/outer_marker_panel_position": outer_pos,
            "extra/outer_marker_panel_rotation_6d": outer_rot6d,
        }

    def get_latest_front_rgb(self) -> Optional[np.ndarray]:
        return self._provider.get_latest_front_rgb()

    def get_latest_outer_marker_seq(self) -> Optional[int]:
        return self._provider.get_latest_outer_marker_seq()

    def get_latest_inner_marker_seq(self) -> Optional[int]:
        return self._provider.get_latest_inner_marker_seq()

    def get_latest_image_seq(self) -> Optional[int]:
        return self._provider.get_latest_image_seq()


def build_rl_task(
    rollout: "RolloutRLPolicy", params: Optional[Mapping[str, object]] = None
):
    if params is None:
        params = {}
    return SingleCardboardCabinetTask(rollout=rollout, params=params)


def run():
    provider = CabinetMarkerPoseProvider()
    provider.start()
    print("[SingleCardboardCabinetTask] CabinetMarkerPoseProvider started.")
    frame_idx = 0
    last_t = time.time()
    try:
        while True:
            frame_idx += 1
            outer_T, outer_seq = provider.get_latest_outer_transform()
            inner_T, inner_seq = provider.get_latest_inner_transform()
            if frame_idx % 20 == 0:
                now = time.time()
                fps = frame_idx / max(now - last_t, 1e-6)
                print(f"[CabinetMarkerPoseProvider] fps~{fps:.1f}", flush=True)
                if outer_T is not None:
                    print(
                        f"[OuterBoard] seq={outer_seq} center={outer_T[:3, 3]} "
                        f"rpy(deg)={_rotmat_to_rpy_deg(outer_T[:3, :3])}",
                        flush=True,
                    )
                if inner_T is not None:
                    print(
                        f"[InnerBoard] ids={SMALL_BOARD_MARKER_IDS} seq={inner_seq} "
                        f"center={inner_T[:3, 3]} rpy(deg)={_rotmat_to_rpy_deg(inner_T[:3, :3])}",
                        flush=True,
                    )
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        provider.stop()
        print("[SingleCardboardCabinetTask] CabinetMarkerPoseProvider stopped.")


if __name__ == "__main__":
    run()
