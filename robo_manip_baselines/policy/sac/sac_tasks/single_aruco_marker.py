from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

import cv2
import numpy as np
from cv2 import aruco

from robo_manip_baselines.policy.rl_policy.rl_tasks.cabinet_marker_detection import (
    FPS,
    transform_from_rvec_tvec,
)

from .single_aruco_marker_policy_state import (
    get_policy_state as get_single_aruco_marker_policy_state,
)

USE_SERIAL = "332522070075"  # neo_front
BASE_CENTER_T_PATH = Path(
    "robo_manip_baselines/calib/base_center_T_neo_front_adjusted.calib"
)
MARKER_ID = 0
MARKER_SIZE_M = 0.0510
MARKER_LOCAL_Z_OFFSET_M = 0.0035
ARUCO_DICT_ID = aruco.DICT_4X4_50
DETECTION_RES_W = 1280
DETECTION_RES_H = 720


def _center_crop_4x3(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    target_w = int(round(h * (4.0 / 3.0)))
    if target_w >= w:
        return img
    x0 = max((w - target_w) // 2, 0)
    return img[:, x0 : x0 + target_w]


def rotation_matrix_to_6d(rotation: np.ndarray) -> np.ndarray:
    x_axis = rotation[:, 0]
    y_axis = rotation[:, 1]
    x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-8)
    y_axis = y_axis - np.dot(x_axis, y_axis) * x_axis
    y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-8)
    return np.concatenate([x_axis, y_axis]).astype(np.float32)


def offset_pose_along_local_z(transform: np.ndarray) -> np.ndarray:
    offset_transform = transform.copy()
    offset_transform[:3, 3] += offset_transform[:3, 2] * MARKER_LOCAL_Z_OFFSET_M
    return offset_transform


def estimate_marker_pose_from_corners(marker_corners, marker_size_m, K, dist_coeffs):
    half = 0.5 * float(marker_size_m)
    object_points = np.array(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float32,
    )
    image_points = np.asarray(marker_corners, dtype=np.float32).reshape(4, 2)
    ok, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        K,
        dist_coeffs,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not ok:
        return None, None
    return rvec.astype(np.float32), tvec.astype(np.float32)


def detect_marker_pose(frame_bgr, detector, K, dist_coeffs):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None or len(ids) == 0:
        return None, None, None, []

    detected_ids = ids.reshape(-1).astype(int).tolist()
    matched = np.where(ids.reshape(-1) == MARKER_ID)[0]
    if matched.size == 0:
        return None, None, None, detected_ids

    marker_corners = corners[int(matched[0])]
    rvec, tvec = estimate_marker_pose_from_corners(
        marker_corners, MARKER_SIZE_M, K, dist_coeffs
    )
    if rvec is None or tvec is None:
        return None, None, None, detected_ids
    return rvec, tvec, marker_corners, detected_ids


def make_detector():
    aruco_dict = aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    params = aruco.DetectorParameters()
    params.cornerRefinementMethod = aruco.CORNER_REFINE_APRILTAG
    params.cornerRefinementWinSize = 5
    params.cornerRefinementMaxIterations = 30
    params.cornerRefinementMinAccuracy = 0.01
    return aruco.ArucoDetector(aruco_dict, params)


def camera_matrix_from_intrinsics(intr):
    K = np.array(
        [[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]],
        dtype=np.float32,
    )
    dist_coeffs = np.array(intr.coeffs[:5], dtype=np.float32)
    return K, dist_coeffs


class ArucoMarkerPoseProvider:
    def __init__(
        self,
        serial: str = USE_SERIAL,
        calib_path: Path = BASE_CENTER_T_PATH,
        res_w: int = DETECTION_RES_W,
        res_h: int = DETECTION_RES_H,
        fps: int = FPS,
    ) -> None:
        self.serial = serial
        self.calib_path = Path(calib_path)
        self.res_w = int(res_w)
        self.res_h = int(res_h)
        self.fps = int(fps)

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._latest_marker_transform: Optional[np.ndarray] = None
        self._latest_front_rgb: Optional[np.ndarray] = None
        self._latest_marker_seq = 0
        self._latest_image_seq = 0
        self._lock = threading.Lock()

        self._detector = make_detector()
        self._base_T_cam = np.loadtxt(self.calib_path).astype(np.float32)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="dual_aruco_marker_pose_provider", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    def get_latest_marker_transform(self) -> Tuple[Optional[np.ndarray], Optional[int]]:
        with self._lock:
            if self._latest_marker_transform is None:
                return None, None
            return self._latest_marker_transform.copy(), self._latest_marker_seq

    def get_latest_front_rgb(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._latest_front_rgb is None:
                return None
            return self._latest_front_rgb.copy()

    def get_latest_marker_seq(self) -> Optional[int]:
        with self._lock:
            return self._latest_marker_seq if self._latest_marker_seq > 0 else None

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
                f"[ArucoMarkerPoseProvider] Failed to start RealSense pipeline: {exc}",
                flush=True,
            )
            return

        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        K, dist_coeffs = camera_matrix_from_intrinsics(color_stream.get_intrinsics())

        try:
            while not self._stop_event.is_set():
                frames = pipeline.wait_for_frames()
                cf = frames.get_color_frame()
                if not cf:
                    continue

                img = np.asanyarray(cf.get_data())
                rgb = cv2.cvtColor(_center_crop_4x3(img), cv2.COLOR_BGR2RGB)
                rgb = cv2.resize(rgb, (640, 480), interpolation=cv2.INTER_AREA)
                with self._lock:
                    self._latest_front_rgb = rgb
                    self._latest_image_seq += 1

                rvec, tvec, _corners, _ids = detect_marker_pose(
                    img, self._detector, K, dist_coeffs
                )
                if rvec is None or tvec is None:
                    continue

                cam_T_marker = transform_from_rvec_tvec(rvec, tvec)
                base_T_marker = offset_pose_along_local_z(
                    self._base_T_cam @ cam_T_marker
                )
                with self._lock:
                    self._latest_marker_transform = base_T_marker.copy()
                    self._latest_marker_seq += 1
        except Exception as exc:
            print(
                f"[ArucoMarkerPoseProvider] Error during detection loop: {exc}",
                flush=True,
            )
        finally:
            pipeline.stop()


@dataclass
class SingleArucoMarkerTask:
    rollout: object
    params: Mapping[str, object] = field(default_factory=dict)
    _provider: ArucoMarkerPoseProvider = field(init=False, repr=False)
    _prev_marker_position: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _prev_marker_rotation_6d: Optional[np.ndarray] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        params = dict(self.params) if isinstance(self.params, Mapping) else {}
        provider_kwargs = {}
        for key in ("serial", "calib_path", "res_w", "res_h", "fps"):
            if key in params:
                provider_kwargs[key] = params[key]
        self._provider = ArucoMarkerPoseProvider(**provider_kwargs)
        self._provider.start()

    def __del__(self) -> None:
        try:
            self._provider.stop()
        except Exception:
            pass

    def on_reset(self) -> None:
        self._prev_marker_position = None
        self._prev_marker_rotation_6d = None

    def get_extra_state(self) -> Dict[str, np.ndarray]:
        marker_T, _marker_seq = self._provider.get_latest_marker_transform()
        if marker_T is None:
            raise RuntimeError(
                f"[SingleArucoMarkerTask] ArUco marker id={MARKER_ID} is not detected yet."
            )
        marker_position = marker_T[:3, 3].astype(np.float32)
        marker_rotation_6d = rotation_matrix_to_6d(marker_T[:3, :3])
        if self._prev_marker_position is None:
            prev_marker_position = marker_position.copy()
            prev_marker_rotation_6d = marker_rotation_6d.copy()
        else:
            prev_marker_position = self._prev_marker_position.copy()
            prev_marker_rotation_6d = self._prev_marker_rotation_6d.copy()

        self._prev_marker_position = marker_position.copy()
        self._prev_marker_rotation_6d = marker_rotation_6d.copy()

        return {
            "marker_position": marker_position,
            "marker_rotation_6d": marker_rotation_6d,
            "prev_marker_position": prev_marker_position,
            "prev_marker_rotation_6d": prev_marker_rotation_6d,
            "trash_bin_position": marker_position,
            "trash_bin_rotation_6d": marker_rotation_6d,
            "prev_trash_bin_position": prev_marker_position,
            "prev_trash_bin_rotation_6d": prev_marker_rotation_6d,
        }

    def get_latest_front_rgb(self) -> Optional[np.ndarray]:
        return self._provider.get_latest_front_rgb()

    def get_latest_marker_seq(self) -> Optional[int]:
        return self._provider.get_latest_marker_seq()

    def get_latest_image_seq(self) -> Optional[int]:
        return self._provider.get_latest_image_seq()

    def get_record_data(self) -> Dict[str, np.ndarray | int]:
        rgb = self.get_latest_front_rgb()
        if rgb is None:
            rgb = np.zeros((480, 640, 3), dtype=np.uint8)

        marker_seq = self.get_latest_marker_seq()
        image_seq = self.get_latest_image_seq()
        return {
            "front_rgb_image": rgb,
            "pose_estimator_seq": -1 if marker_seq is None else int(marker_seq),
            "image_seq": -1 if image_seq is None else int(image_seq),
        }

    def get_policy_state(self):
        return get_single_aruco_marker_policy_state(self, self.rollout)


def build_ppo_task(rollout, params: Optional[Mapping[str, object]] = None):
    if params is None:
        params = {}
    return SingleArucoMarkerTask(rollout=rollout, params=params)


def build_rl_task(rollout, params: Optional[Mapping[str, object]] = None):
    return build_ppo_task(rollout, params)


def put_line(img, text, y, color=(235, 235, 235), scale=0.54):
    cv2.putText(
        img,
        text,
        (12, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        2,
        cv2.LINE_AA,
    )
    return y + 24


def project_marker_center(img, K, dist_coeffs, rvec, tvec, color, label):
    center_px, _ = cv2.projectPoints(
        np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        rvec,
        tvec,
        K,
        dist_coeffs,
    )
    cx, cy = center_px[0, 0].astype(int)
    cv2.circle(img, (cx, cy), 8, color, -1)
    cv2.putText(
        img,
        label,
        (cx + 10, cy - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA,
    )


def run_viewer():
    import pyrealsense2 as rs

    base_T_cam = np.loadtxt(Path(BASE_CENTER_T_PATH)).astype(np.float32)
    detector = make_detector()

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(USE_SERIAL)
    config.enable_stream(
        rs.stream.color, DETECTION_RES_W, DETECTION_RES_H, rs.format.bgr8, FPS
    )
    profile = pipeline.start(config)

    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    K, dist_coeffs = camera_matrix_from_intrinsics(color_stream.get_intrinsics())
    cam_T_base = np.linalg.inv(base_T_cam)
    R_base = cam_T_base[:3, :3]
    base_rvec, _ = cv2.Rodrigues(R_base)
    base_tvec = cam_T_base[:3, 3].reshape(3, 1)

    print(f"base_T_cam: {Path(BASE_CENTER_T_PATH)}")
    print(f"marker id={MARKER_ID}, size={MARKER_SIZE_M} m, dict=DICT_4X4_50")

    t_prev = time.time()
    fps_est = 0.0
    last_print = 0.0

    try:
        while True:
            frames = pipeline.wait_for_frames()
            cf = frames.get_color_frame()
            if not cf:
                continue

            img = np.asanyarray(cf.get_data())
            rvec, tvec, marker_corners, detected_ids = detect_marker_pose(
                img, detector, K, dist_coeffs
            )

            marker_base = None
            marker_rotation_6d = None
            if marker_corners is not None:
                aruco.drawDetectedMarkers(
                    img,
                    [marker_corners],
                    np.array([[MARKER_ID]], dtype=np.int32),
                )
                cam_T_marker = transform_from_rvec_tvec(rvec, tvec)
                base_T_marker = offset_pose_along_local_z(base_T_cam @ cam_T_marker)

                marker_base = base_T_marker[:3, 3]
                marker_rotation_6d = rotation_matrix_to_6d(base_T_marker[:3, :3])
                project_marker_center(
                    img,
                    K,
                    dist_coeffs,
                    rvec,
                    tvec,
                    (255, 0, 255),
                    f"marker {MARKER_ID}",
                )
                cv2.drawFrameAxes(
                    img,
                    K,
                    dist_coeffs,
                    rvec,
                    tvec,
                    MARKER_SIZE_M * 0.5,
                    3,
                )
            cv2.drawFrameAxes(
                img,
                K,
                dist_coeffs,
                base_rvec,
                base_tvec,
                0.20,
                4,
            )
            now = time.time()
            fps_est = 0.9 * fps_est + 0.1 * (1.0 / max(1e-6, now - t_prev))
            t_prev = now

            overlay = img.copy()
            panel_w = 620
            cv2.rectangle(overlay, (0, 0), (panel_w, DETECTION_RES_H), (0, 0, 0), -1)
            img = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)
            y = 30
            y = put_line(img, f"Marker {MARKER_ID} viewer  FPS: {fps_est:.1f}", y)
            y = put_line(img, f"Serial: {USE_SERIAL}", y, (220, 220, 220))
            y = put_line(img, f"Marker size: {MARKER_SIZE_M:.5f} m", y)
            y = put_line(img, f"base_T_cam: {Path(BASE_CENTER_T_PATH)}", y)
            y = put_line(img, f"Detected IDs: {detected_ids}", y)
            y += 8
            y = put_line(
                img,
                f"Marker {MARKER_ID}: {'OK' if marker_base is not None else '---'}",
                y,
                (255, 160, 255) if marker_base is not None else (180, 180, 180),
            )
            if marker_base is not None:
                y = put_line(img, f"  base position: {np.round(marker_base, 4)}", y)
                y = put_line(
                    img,
                    f"  policy rotation_6d: {np.round(marker_rotation_6d, 4)}",
                    y,
                )

            if time.time() - last_print > 1.0:
                if marker_base is not None:
                    print(
                        f"[Marker{MARKER_ID}] base_position={marker_base} "
                        f"policy_rotation_6d={marker_rotation_6d}",
                        flush=True,
                    )
                else:
                    print(
                        f"[Marker{MARKER_ID}] not detected. detected_ids={detected_ids}",
                        flush=True,
                    )
                last_print = time.time()

            cv2.imshow(f"Marker {MARKER_ID} Viewer", img)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_viewer()
