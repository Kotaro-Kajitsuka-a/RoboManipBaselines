import inspect
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
_APRILTAG_SRC = _REPO_ROOT / "external" / "check_AprilTag" / "src"
if _APRILTAG_SRC.exists():
    apriltag_src_str = str(_APRILTAG_SRC)
    if apriltag_src_str not in sys.path:  # pragma: no cover - guard
        sys.path.append(apriltag_src_str)

try:
    from pose_viewer import (
        build_detector,
        build_homogeneous_transform,
        solve_tag_poses,
    )
except ImportError:  # pragma: no cover - optional dependency
    build_detector = None  # type: ignore[assignment]
    build_homogeneous_transform = None  # type: ignore[assignment]
    solve_tag_poses = None  # type: ignore[assignment]


DEFAULT_TAG_SIZE_M = 0.0309
DEFAULT_DETECTOR_THREADS = 4
DEFAULT_DETECTOR_DECIMATE = 1.0
DEFAULT_DETECTOR_SIGMA = 1.0
DEFAULT_DETECTOR_SHARPENING = 0.1
DEFAULT_CAMERA_NAME = "front"


def load_base_to_camera_transform(path: Path) -> Optional[np.ndarray]:
    try:
        matrix = np.loadtxt(path, delimiter=",", dtype=np.float64)
    except FileNotFoundError:
        print(
            f"[MarkerDetection] T_base→camera transform not found at {path}. "
            "Marker detection will be disabled.",
            flush=True,
        )
        return None
    except Exception as exc:  # pragma: no cover - I/O error
        print(
            f"[MarkerDetection] Failed to load T_base→camera transform from {path}: {exc}",
            flush=True,
        )
        return None

    matrix = matrix.reshape(4, 4)
    return matrix.astype(np.float64)


def extract_camera_intrinsic_info(camera) -> Optional[Dict[str, Any]]:
    if camera is None:
        return None

    info: Dict[str, Any] = {}
    candidate_attrs = (
        "color_intrinsics",
        "intrinsics",
        "color_intrinsic",
        "intrinsic",
    )
    for attr in candidate_attrs:
        intr = getattr(camera, attr, None)
        if intr is None:
            continue

        def _get_value(name: str):
            if hasattr(intr, name):
                return getattr(intr, name)
            if isinstance(intr, dict):
                return intr.get(name)
            if hasattr(intr, "__getitem__"):
                try:
                    return intr[name]
                except Exception:
                    return None
            return None

        fx = _get_value("fx")
        fy = _get_value("fy")
        ppx = _get_value("ppx")
        ppy = _get_value("ppy")
        coeffs = _get_value("coeffs")

        if fx is not None:
            info["fx"] = float(fx)
        if fy is not None:
            info["fy"] = float(fy)
        if ppx is not None:
            info["ppx"] = float(ppx)
        if ppy is not None:
            info["ppy"] = float(ppy)
        if coeffs is not None:
            info["coeffs"] = list(coeffs) if not isinstance(coeffs, list) else coeffs

        if info:
            break

    color_fovy = getattr(camera, "color_fovy", None)
    if color_fovy is not None:
        info["color_fovy"] = float(color_fovy)

    frame_width = getattr(camera, "color_width", None) or getattr(camera, "width", None)
    frame_height = getattr(camera, "color_height", None) or getattr(camera, "height", None)
    if frame_width is not None:
        info["frame_width"] = int(frame_width)
    if frame_height is not None:
        info["frame_height"] = int(frame_height)

    required_keys = ("fx", "fy", "ppx", "ppy")
    if not all(key in info for key in required_keys):
        if hasattr(camera, "_pipeline"):
            try:
                import pyrealsense2 as rs  # type: ignore

                frames = camera._pipeline.wait_for_frames()  # noqa: SLF001
                color_profile = frames.get_color_frame().profile.as_video_stream_profile()
                intr = color_profile.intrinsics
                coeffs = list(intr.coeffs[:5])
                info.update(
                    {
                        "fx": float(intr.fx),
                        "fy": float(intr.fy),
                        "ppx": float(intr.ppx),
                        "ppy": float(intr.ppy),
                        "coeffs": coeffs,
                        "frame_width": intr.width,
                        "frame_height": intr.height,
                    }
                )
                setattr(camera, "color_fx", intr.fx)
                setattr(camera, "color_fy", intr.fy)
                setattr(camera, "color_ppx", intr.ppx)
                setattr(camera, "color_ppy", intr.ppy)
                setattr(camera, "color_coeffs", coeffs)
                setattr(camera, "color_width", intr.width)
                setattr(camera, "color_height", intr.height)
            except Exception:
                pass

    if not all(key in info for key in required_keys):
        raise RuntimeError(
            "[MarkerDetection] Camera intrinsics (fx, fy, ppx, ppy) are unavailable. "
            "Ensure the RealSense intrinsics are accessible before starting the rollout."
        )

    return info


class FrontCameraDetectionWorker:
    """Process front-camera frames on a background thread to estimate AprilTag poses."""

    def __init__(
        self,
        base_to_camera: Optional[np.ndarray],
        intrinsic_info: Optional[Dict[str, Any]] = None,
        tag_size_m: float = DEFAULT_TAG_SIZE_M,
    ):
        self._base_to_camera = None if base_to_camera is None else base_to_camera.astype(np.float64)
        self._intrinsic_info: Dict[str, Any] = intrinsic_info or {}
        self._tag_size_m = float(tag_size_m)
        self._frame_queue: "queue.Queue[Optional[Tuple[np.ndarray, float]]]" = queue.Queue(maxsize=1)
        self._result_queue: "queue.Queue[Tuple[float, Dict[int, np.ndarray]]]" = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()
        self._processing_thread: Optional[threading.Thread] = None
        self._latest_lock = threading.Lock()
        self._latest_gray: Optional[np.ndarray] = None
        self._latest_timestamp: Optional[float] = None
        self._latest_transforms: Dict[int, np.ndarray] = {}
        self._latest_transform_times: Dict[int, float] = {}
        self._latest_transforms_timestamp: Optional[float] = None
        self._detector = None
        self._camera_matrix: Optional[np.ndarray] = None
        self._dist_coeffs: Optional[np.ndarray] = None
        self._detection_available = (
            build_detector is not None
            and build_homogeneous_transform is not None
            and solve_tag_poses is not None
            and self._base_to_camera is not None
        )
        self._frame_counter = 0
        self._last_detection_count = None

    def start(self):
        if self._processing_thread and self._processing_thread.is_alive():
            return

        if not self._detection_available:
            print(
                "[FrontCameraDetectionWorker] Marker detection disabled (missing dependencies or calibration).",
                flush=True,
            )

        detector = self._get_or_build_detector()
        if detector is None:
            self._detection_available = False
            print(
                "[FrontCameraDetectionWorker] Detector initialization failed; detection disabled.",
                flush=True,
            )

        self._stop_event.clear()
        self._processing_thread = threading.Thread(
            target=self._processing_loop,
            name="front_camera_detection",
            daemon=True,
        )
        self._processing_thread.start()

    def stop(self):
        self._stop_event.set()
        try:
            self._frame_queue.put_nowait(None)
        except queue.Full:
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._frame_queue.put_nowait(None)
            except queue.Full:
                pass

        if self._processing_thread:
            self._processing_thread.join(timeout=1.0)
            self._processing_thread = None

        while True:
            try:
                self._result_queue.get_nowait()
            except queue.Empty:
                break


    def submit_frame(self, rgb_image: np.ndarray) -> None:
        """Queue a frame for detection."""
        if rgb_image is None or self._stop_event.is_set():
            return

        gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        timestamp = time.time()

        with self._latest_lock:
            self._latest_gray = gray_image
            self._latest_timestamp = timestamp

        payload = (gray_image, timestamp)
        try:
            self._frame_queue.put_nowait(payload)
        except queue.Full:
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._frame_queue.put_nowait(payload)
            except queue.Full:
                pass
        self._frame_counter += 1

    def get_latest_frame(self) -> Tuple[Optional[np.ndarray], Optional[float]]:
        with self._latest_lock:
            if self._latest_gray is None:
                return None, None
            return self._latest_gray.copy(), self._latest_timestamp

    def get_latest_transforms(self) -> Tuple[Dict[int, np.ndarray], Optional[float]]:
        with self._latest_lock:
            return (
                {tag_id: matrix.copy() for tag_id, matrix in self._latest_transforms.items()},
                self._latest_transforms_timestamp,
            )

    def get_latest_transform_times(self) -> Dict[int, float]:
        with self._latest_lock:
            return dict(self._latest_transform_times)

    def poll_transforms(self) -> Tuple[Optional[Dict[int, np.ndarray]], Optional[float]]:
        try:
            timestamp, transforms = self._result_queue.get_nowait()
        except queue.Empty:
            return None, None

        copied = {tag_id: matrix.copy() for tag_id, matrix in transforms.items()}
        with self._latest_lock:
            self._latest_transforms = {
                tag_id: matrix.copy() for tag_id, matrix in copied.items()
            }
            if timestamp is not None:
                self._latest_transforms_timestamp = timestamp
        return copied, timestamp

    def _processing_loop(self):
        while not self._stop_event.is_set():
            try:
                item = self._frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if item is None:
                break

            gray_image, timestamp = item
            transforms: Dict[int, np.ndarray] = {}

            if self._detection_available:
                detector = self._get_or_build_detector()
                camera_mats = self._ensure_camera_parameters(gray_image)
                if detector is not None and camera_mats is not None:
                    K, dist_coeffs = camera_mats
                    try:
                        poses = solve_tag_poses(
                            detector,
                            gray_image,
                            K,
                            dist_coeffs,
                            self._tag_size_m,
                        )
                    except Exception:
                        poses = []
                    if self._last_detection_count != len(poses):
                        print(
                            f"[FrontCameraDetectionWorker] Detected {len(poses)} tags in current frame.",
                            flush=True,
                        )
                        self._last_detection_count = len(poses)

                    for pose in poses:
                        if int(pose.tag_id) == 3:
                            continue
                        try:
                            T_cam_to_tag = build_homogeneous_transform(pose.rvec, pose.tvec)
                        except Exception:
                            continue
                        T_base_to_tag = self._base_to_camera @ T_cam_to_tag
                        transforms[int(pose.tag_id)] = T_base_to_tag

            payload = None
            with self._latest_lock:
                if transforms:
                    for tag_id, matrix in transforms.items():
                        self._latest_transforms[tag_id] = matrix.copy()
                        self._latest_transform_times[tag_id] = timestamp
                    self._latest_transforms_timestamp = timestamp
                    combined = {
                        tag_id: matrix.copy()
                        for tag_id, matrix in self._latest_transforms.items()
                    }
                    payload = (timestamp, combined)

            if payload is not None:
                try:
                    self._result_queue.put_nowait(payload)
                except queue.Full:
                    try:
                        self._result_queue.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        self._result_queue.put_nowait(payload)
                    except queue.Full:
                        pass

    def _get_or_build_detector(self):
        if self._detector is not None:
            return self._detector
        if build_detector is None:
            return None

        detector_kwargs = {
            "nthreads": DEFAULT_DETECTOR_THREADS,
            "quad_decimate": DEFAULT_DETECTOR_DECIMATE,
            "quad_sigma": DEFAULT_DETECTOR_SIGMA,
            "refine_edges": True,
            "decode_sharpening": DEFAULT_DETECTOR_SHARPENING,
        }
        try:
            sig = inspect.signature(build_detector)
            accepted = {
                key: value for key, value in detector_kwargs.items() if key in sig.parameters
            }
        except (TypeError, ValueError):  # pragma: no cover - signature introspection failure
            accepted = detector_kwargs

        try:
            self._detector = build_detector("tag36h11", **accepted)
        except TypeError:
            try:
                self._detector = build_detector("tag36h11")
            except Exception:
                self._detector = None
        except Exception:  # pragma: no cover - detector creation failure
            self._detector = None
        else:
            if self._detector is not None:
                print(
                    "[FrontCameraDetectionWorker] AprilTag detector initialized (tag36h11).",
                    flush=True,
                )
        return self._detector

    # Backward compatibility: legacy callers may still reference the old name.
    def _build_detector(self):
        return self._get_or_build_detector()

    def _ensure_camera_parameters(
        self, frame: np.ndarray
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if self._camera_matrix is not None and self._dist_coeffs is not None:
            return self._camera_matrix, self._dist_coeffs

        height, width = frame.shape[:2]
        info = self._intrinsic_info
        fx = info.get("fx")
        fy = info.get("fy")
        ppx = info.get("ppx")
        ppy = info.get("ppy")
        coeffs = info.get("coeffs")

        if fx is not None and fy is not None and ppx is not None and ppy is not None:
            K = np.array(
                [
                    [float(fx), 0.0, float(ppx)],
                    [0.0, float(fy), float(ppy)],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
            dist_coeffs = (
                np.array(coeffs[:5], dtype=np.float32)
                if isinstance(coeffs, (list, tuple, np.ndarray))
                else np.zeros(5, dtype=np.float32)
            )
            self._camera_matrix = K
            self._dist_coeffs = dist_coeffs
            return K, dist_coeffs

        fovy_deg = info.get("color_fovy")
        frame_w = int(info.get("frame_width", width))
        frame_h = int(info.get("frame_height", height))
        if fovy_deg is not None:
            fovy_rad = np.deg2rad(float(fovy_deg))
            fy = (frame_h / 2.0) / np.tan(max(1e-6, fovy_rad / 2.0))
            fy = float(fy)
            fx = fy * (frame_w / max(frame_h, 1))
            cx = frame_w / 2.0
            cy = frame_h / 2.0
            K = np.array(
                [
                    [fx, 0.0, cx],
                    [0.0, fy, cy],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
            self._camera_matrix = K
            self._dist_coeffs = np.zeros(5, dtype=np.float32)
            return self._camera_matrix, self._dist_coeffs

        fx = fy = max(frame_w, frame_h)
        cx = frame_w / 2.0
        cy = frame_h / 2.0
        self._camera_matrix = np.array(
            [
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        self._dist_coeffs = np.zeros(5, dtype=np.float32)
        return self._camera_matrix, self._dist_coeffs


class MarkerManager:
    """Manage marker config, detection worker lifecycle, and cached transforms."""

    def __init__(
        self,
        base_to_camera: Optional[np.ndarray],
        marker_definitions: Optional[List[Dict[str, Any]]] = None,
        marker_camera_names: Optional[List[str]] = None,
        default_tag_size_m: float = DEFAULT_TAG_SIZE_M,
    ):
        self.base_to_camera = None if base_to_camera is None else base_to_camera.astype(np.float64)
        self.marker_definitions: List[Dict[str, Any]] = marker_definitions or []
        self.required_marker_ids: List[int] = [
            entry["id"] for entry in self.marker_definitions if "id" in entry
        ]
        self.marker_name_map: Dict[int, str] = {
            entry["id"]: entry.get("name", f"marker_{entry['id']}")  # type: ignore[index]
            for entry in self.marker_definitions
            if "id" in entry
        }
        self.marker_size_map: Dict[int, float] = {
            entry["id"]: float(entry.get("size_m", DEFAULT_TAG_SIZE_M))  # type: ignore[index]
            for entry in self.marker_definitions
            if "id" in entry
        }
        self.marker_camera_names: List[str] = (
            [str(name) for name in marker_camera_names] if marker_camera_names else [DEFAULT_CAMERA_NAME]
        )
        self._tag_size_m = float(
            self.marker_size_map[self.required_marker_ids[0]]
        ) if self.required_marker_ids else float(default_tag_size_m)

        self._worker: Optional[FrontCameraDetectionWorker] = None
        self._active_camera: Optional[str] = None
        self.marker_transform_cache: Dict[int, np.ndarray] = {}
        self._last_missing_ids: Optional[List[int]] = None

    def stop(self):
        if self._worker is not None:
            self._worker.stop()
            self._worker = None
        self._active_camera = None

    def reset_cache(self):
        self.marker_transform_cache.clear()
        self._last_missing_ids = None

    def start_with_cameras(
        self, primary_cameras: Dict[str, Any], backup_cameras: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Try to start detection using the first available camera."""
        backup_cameras = backup_cameras or {}
        if self.base_to_camera is None:
            return False

        for camera_name in self.marker_camera_names:
            camera_obj = primary_cameras.get(camera_name) or backup_cameras.get(camera_name)
            if camera_obj is None:
                continue
            intrinsic_info = extract_camera_intrinsic_info(camera_obj)
            worker = FrontCameraDetectionWorker(
                base_to_camera=self.base_to_camera,
                intrinsic_info=intrinsic_info,
                tag_size_m=self._tag_size_m,
            )
            worker.start()
            self._worker = worker
            self._active_camera = camera_name
            return True
        return False

    def submit_rgb_images(self, rgb_images: Optional[Dict[str, np.ndarray]]) -> bool:
        """Submit a frame from rgb_images to the worker."""
        if self._worker is None or rgb_images is None:
            return False
        candidate_names = (
            [self._active_camera] if self._active_camera is not None else self.marker_camera_names
        )
        for camera_name in candidate_names:
            if camera_name is None:
                continue
            frame = rgb_images.get(camera_name)
            if frame is None:
                continue
            self._worker.submit_frame(frame.copy())
            return True
        return False

    def get_latest_frame(self):
        if self._worker is None:
            return None, None
        return self._worker.get_latest_frame()

    def get_latest_transforms(self, poll: bool = False):
        """Get latest transforms from the worker."""
        if self._worker is None:
            return None, None
        if poll:
            transforms, timestamp = self._worker.poll_transforms()
            if transforms is not None:
                return transforms, timestamp
        return self._worker.get_latest_transforms()

    def refresh_cache(self, poll: bool = False) -> Optional[float]:
        """Update cache from worker results, returning latest timestamp."""
        transforms, timestamp = self.get_latest_transforms(poll=poll)
        if not transforms:
            transforms, timestamp = self.get_latest_transforms()
        if transforms:
            for marker_id, matrix in transforms.items():
                self.marker_transform_cache[marker_id] = matrix.copy()
        return timestamp

    def missing_required_ids(self) -> List[int]:
        """Return missing required marker IDs based on cached transforms."""
        if not self.required_marker_ids:
            return []
        return [
            marker_id
            for marker_id in self.required_marker_ids
            if marker_id not in self.marker_transform_cache
        ]

    def warn_on_missing(self):
        missing_now = self.missing_required_ids()
        if missing_now != self._last_missing_ids:
            if missing_now:
                print(
                    f"[MarkerManager] Warning: Missing cached transforms for marker IDs {missing_now}.",
                    flush=True,
                )
            self._last_missing_ids = missing_now
