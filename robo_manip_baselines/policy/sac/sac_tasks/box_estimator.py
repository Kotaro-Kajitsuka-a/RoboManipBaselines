from __future__ import annotations

import threading
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Tuple

import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

from .dual_box_rotation import (
    BASE_CENTER_T_PATH,
    BOX_DEPTH_M,
    GRIDBOARD_H_M,
    GRIDBOARD_W_M,
    RES_H,
    RES_W,
    FPS,
    USE_SERIAL,
    _center_crop_4x3,
    _map_points_to_record_frame,
)
from .dual_box_rotation import DualBoxRotationTask

KEYPOINT_ORDER = [0, 1, 2, 3]


def _solve_pnp_from_keypoints(keypoints_xy, K, dist_coeffs):
    if keypoints_xy is None or len(keypoints_xy) < 4:
        return None

    img_pts = np.asarray([keypoints_xy[i] for i in KEYPOINT_ORDER], dtype=np.float32).reshape(-1, 2)
    obj_pts = np.array(
        [
            [0.0, 0.0, 0.0],
            [GRIDBOARD_W_M, 0.0, 0.0],
            [GRIDBOARD_W_M, GRIDBOARD_H_M, 0.0],
            [0.0, GRIDBOARD_H_M, 0.0],
        ],
        dtype=np.float32,
    )

    ok, rvec, tvec = cv2.solvePnP(
        obj_pts,
        img_pts,
        K,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None
    return rvec, tvec


class YoloBoxPoseProvider:
    def __init__(
        self,
        pt_path: str,
        serial: str = USE_SERIAL,
        calib_path=BASE_CENTER_T_PATH,
        res_w: int = RES_W,
        res_h: int = RES_H,
        fps: int = FPS,
    ) -> None:
        if not pt_path:
            raise ValueError("pt_path must be a non-empty string.")

        self.serial = serial
        self.res_w = int(res_w)
        self.res_h = int(res_h)
        self.fps = int(fps)
        self._base_T_cam = np.loadtxt(Path(calib_path).expanduser().resolve()).astype(np.float32)

        self._model = YOLO(pt_path)

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._latest_transform: Optional[np.ndarray] = None
        self._latest_timestamp: Optional[float] = None
        self._latest_front_rgb: Optional[np.ndarray] = None
        self._latest_board_corners: Optional[np.ndarray] = None
        self._latest_box_pose_seq: int = 0
        self._latest_image_seq: int = 0
        self._latest_board_seq: int = 0
        self._lock = threading.Lock()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="yolo_box_pose_provider", daemon=True
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
            return self._latest_transform.copy(), self._latest_timestamp

    def get_latest_front_rgb(self) -> Tuple[Optional[np.ndarray], Optional[float]]:
        with self._lock:
            if self._latest_front_rgb is None:
                return None, None
            return self._latest_front_rgb, self._latest_timestamp

    def get_latest_box_pose_seq(self) -> Optional[int]:
        with self._lock:
            return self._latest_box_pose_seq if self._latest_box_pose_seq > 0 else None

    def get_latest_image_seq(self) -> Optional[int]:
        with self._lock:
            return self._latest_image_seq if self._latest_image_seq > 0 else None

    def get_latest_board_corners(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._latest_board_corners is None:
                return None
            return self._latest_board_corners.copy()

    def _run(self) -> None:
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(self.serial)
        config.enable_stream(rs.stream.color, self.res_w, self.res_h, rs.format.rgb8, self.fps)

        try:
            profile = pipeline.start(config)
        except Exception as exc:
            print(f"[YoloBoxPoseProvider] Failed to start RealSense pipeline: {exc}", flush=True)
            return

        # Match normal record-data camera behavior (no manual exposure/gain override).
        # Keep rgb8 stream + 1280x720 capture path for SAC-specific cropping/resizing.

        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_stream.get_intrinsics()
        fx, fy, cx, cy = intr.fx, intr.fy, intr.ppx, intr.ppy
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        dist_coeffs = np.array(intr.coeffs[:5], dtype=np.float32)

        R_flip_x = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32)
        center_offset = np.array([GRIDBOARD_W_M * 0.5, GRIDBOARD_H_M * 0.5, 0.0], dtype=np.float32)
        z_offset = np.array([0.0, 0.0, -BOX_DEPTH_M * 0.5], dtype=np.float32)

        board_corners_3d = np.array(
            [
                [0.0, 0.0, 0.0],
                [GRIDBOARD_W_M, 0.0, 0.0],
                [GRIDBOARD_W_M, GRIDBOARD_H_M, 0.0],
                [0.0, GRIDBOARD_H_M, 0.0],
            ],
            dtype=np.float32,
        )

        try:
            while not self._stop_event.is_set():
                frames = pipeline.wait_for_frames()
                cf = frames.get_color_frame()
                if not cf:
                    continue

                img = np.asanyarray(cf.get_data())  # RGB
                bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                rgb = _center_crop_4x3(img)
                rgb = cv2.resize(rgb, (640, 480), interpolation=cv2.INTER_AREA)
                board_corners_record = np.full((4, 2), -1.0, dtype=np.float32)

                results = self._model.track(bgr, persist=True, verbose=False)
                keypoints = results[0].keypoints if results else None
                if keypoints is None or keypoints.xy is None or len(keypoints.xy) == 0:
                    with self._lock:
                        self._latest_front_rgb = rgb
                        self._latest_image_seq += 1
                        self._latest_board_corners = board_corners_record
                        self._latest_board_seq += 1
                    continue

                pts = keypoints.xy[0].detach().cpu().numpy()
                pnp = _solve_pnp_from_keypoints(pts, K, dist_coeffs)
                if pnp is None:
                    with self._lock:
                        self._latest_front_rgb = rgb
                        self._latest_image_seq += 1
                        self._latest_board_corners = board_corners_record
                        self._latest_board_seq += 1
                    continue

                rvec, tvec = pnp
                proj, _ = cv2.projectPoints(board_corners_3d, rvec, tvec, K, dist_coeffs)
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
        except Exception as exc:
            print(f"[YoloBoxPoseProvider] Error during detection loop: {exc}", flush=True)
        finally:
            pipeline.stop()


@dataclass
class YoloBoxRotationTask(DualBoxRotationTask):
    rollout: "RolloutSac"
    params: Mapping[str, object] = field(default_factory=dict)
    _provider: YoloBoxPoseProvider = field(init=False, repr=False)

    def __post_init__(self) -> None:
        params_dict = dict(self.params) if isinstance(self.params, Mapping) else {}
        pt_path = params_dict.get("pt_path")
        if not pt_path:
            raise ValueError("pt_path must be provided for YoloBoxRotationTask.")

        provider_kwargs = {"pt_path": str(pt_path)}
        for key in ("serial", "calib_path", "res_w", "res_h", "fps"):
            if key in params_dict:
                provider_kwargs[key] = params_dict[key]

        camera_name = str(params_dict.get("camera_name", "")).lower()
        if camera_name == "top":
            provider_kwargs.setdefault("serial", USE_SERIAL)
            provider_kwargs.setdefault("calib_path", BASE_CENTER_T_PATH)

        self._provider = YoloBoxPoseProvider(**provider_kwargs)
        self._provider.start()
        self._pushpoint_local_right = None
        self._pushpoint_local_left = None
        self._last_box_pose = None

    def get_latest_top_rgb(self) -> Optional[np.ndarray]:
        return self.get_latest_front_rgb()

    def get_latest_top_aruco_board_corners(self) -> Optional[np.ndarray]:
        return self.get_latest_front_aruco_board_corners()


def build_ppo_task(rollout: "RolloutSac", params: Optional[Mapping[str, object]] = None):
    if params is None:
        params = {}
    return YoloBoxRotationTask(rollout=rollout, params=params)
