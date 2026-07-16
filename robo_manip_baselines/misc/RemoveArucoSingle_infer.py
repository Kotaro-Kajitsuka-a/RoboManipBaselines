import argparse
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
from cv2 import aruco


MARKER_ID = 0
ARUCO_DICT_ID = aruco.DICT_4X4_50
MARKER_EXPAND_SCALE = 1.55
MAX_MARKER_JUMP_PX = 80.0


def _aruco_dict(dict_id: int):
    try:
        return aruco.getPredefinedDictionary(dict_id)
    except AttributeError:
        return aruco.Dictionary_get(dict_id)


def _detector_parameters():
    try:
        params = aruco.DetectorParameters_create()
    except AttributeError:
        params = aruco.DetectorParameters()

    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 53
    params.adaptiveThreshWinSizeStep = 4
    params.minMarkerPerimeterRate = 0.01
    params.maxMarkerPerimeterRate = 4.0
    params.polygonalApproxAccuracyRate = 0.05
    params.errorCorrectionRate = 0.8
    if hasattr(aruco, "CORNER_REFINE_SUBPIX"):
        params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
        params.cornerRefinementWinSize = 5
        params.cornerRefinementMaxIterations = 30
        params.cornerRefinementMinAccuracy = 0.01
    return params


def _gray_image(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def _candidate_gray_images(image: np.ndarray, scales: tuple[float, ...]):
    gray = _gray_image(image)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    contrast = clahe.apply(gray)
    blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
    sharpened = cv2.addWeighted(gray, 1.7, blur, -0.7, 0.0)
    base_images = (gray, contrast, sharpened)

    for scale in scales:
        for candidate in base_images:
            if scale == 1.0:
                yield candidate, scale
                continue
            height, width = candidate.shape[:2]
            resized = cv2.resize(
                candidate,
                (int(width * scale), int(height * scale)),
                interpolation=cv2.INTER_CUBIC,
            )
            yield resized, scale


def detect_marker_corners(
    image: np.ndarray,
    marker_id: int,
    aruco_dict,
    parameters,
    scales: tuple[float, ...] = (1.0, 1.5, 2.0),
) -> np.ndarray | None:
    for candidate, scale in _candidate_gray_images(image, scales):
        corners, ids, _ = aruco.detectMarkers(
            candidate, aruco_dict, parameters=parameters
        )
        if ids is None or len(ids) == 0:
            continue

        ids_flat = ids.reshape(-1)
        matched = np.where(ids_flat == marker_id)[0]
        if matched.size == 0:
            continue

        marker_corners = np.asarray(corners[int(matched[0])], dtype=np.float32).reshape(
            4, 2
        )
        if scale != 1.0:
            marker_corners = marker_corners / scale
        return marker_corners.astype(np.float32)
    return None


def _is_jump_outlier(
    corners: np.ndarray,
    prev_corners: np.ndarray | None,
    max_jump_px: float,
) -> bool:
    if prev_corners is None:
        return False
    deltas = np.linalg.norm(corners - prev_corners, axis=-1)
    return bool(np.any(deltas > max_jump_px))


def track_corners(
    prev_frame: np.ndarray,
    frame: np.ndarray,
    prev_corners: np.ndarray,
    max_point_move_px: float,
) -> np.ndarray | None:
    prev_gray = _gray_image(prev_frame)
    gray = _gray_image(frame)
    prev_points = np.asarray(prev_corners, dtype=np.float32).reshape(-1, 1, 2)
    points, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        gray,
        prev_points,
        None,
        winSize=(31, 31),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    if points is None or status is None or not np.all(status):
        return None

    tracked = points.reshape(4, 2)
    deltas = np.linalg.norm(tracked - prev_points.reshape(4, 2), axis=1)
    if np.any(deltas > max_point_move_px):
        return None
    return tracked.astype(np.float32)


def select_corners_with_fallback(
    raw_corners: np.ndarray | None,
    last_good_corners: np.ndarray | None,
    prev_frame: np.ndarray | None,
    frame: np.ndarray,
    max_jump_px: float,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if raw_corners is not None:
        raw_corners = np.asarray(raw_corners, dtype=np.float32).reshape(4, 2)
        if not _is_jump_outlier(raw_corners, last_good_corners, max_jump_px):
            return raw_corners, raw_corners.copy()

    if prev_frame is not None and last_good_corners is not None:
        tracked = track_corners(prev_frame, frame, last_good_corners, max_jump_px)
        if tracked is not None:
            return tracked, tracked.copy()

    if last_good_corners is not None:
        return last_good_corners.copy(), last_good_corners
    return None, last_good_corners


def expand_corners(corners: np.ndarray, expand_scale: float) -> np.ndarray:
    corners = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    if expand_scale < 1.0:
        raise ValueError(f"expand_scale must be >= 1.0, got {expand_scale}")
    center = corners.mean(axis=0)
    return center + (corners - center) * float(expand_scale)


def build_marker_mask(
    shape: tuple[int, int],
    corners: np.ndarray,
    expand_scale: float,
    scale: int = 4,
) -> np.ndarray:
    height, width = shape
    expanded = expand_corners(corners, expand_scale)
    scale = max(2, int(scale))
    mask = np.zeros((height * scale, width * scale), dtype=np.uint8)
    pts = (expanded * scale).astype(np.int32).reshape(1, 4, 2)
    cv2.fillPoly(mask, [pts], color=255)
    return cv2.resize(mask, (width, height), interpolation=cv2.INTER_AREA)


def remove_marker(frame: np.ndarray, corners: np.ndarray):
    mask = build_marker_mask(frame.shape[:2], corners, MARKER_EXPAND_SCALE)
    valid_mask = np.where(mask > 0, 0, 255).astype(np.uint8)
    output = np.empty_like(frame)
    cv2.xphoto.inpaint(frame, valid_mask, output, cv2.xphoto.INPAINT_FSR_FAST)
    return output


def iter_target_mp4(input_dir: Path, camera_name: str):
    target_name = f"{camera_name}_rgb_image.rmb.mp4"
    for mp4_path in input_dir.rglob(target_name):
        if ".bedrooms" in mp4_path.parts:
            continue
        yield mp4_path


def copy_dataset_dir(input_dir: Path) -> Path:
    if not input_dir.exists():
        raise FileNotFoundError(f"input_dir not found: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"input_dir is not a directory: {input_dir}")

    output_dir = input_dir.with_name(f"{input_dir.name}_single_marker_removed")
    if output_dir.exists():
        raise FileExistsError(f"output_dir already exists: {output_dir}")

    def ignore_hidden_dirs(dir_path: str, entries: list[str]) -> list[str]:
        ignored: list[str] = []
        for name in entries:
            if name.startswith(".") and (Path(dir_path) / name).is_dir():
                ignored.append(name)
        return ignored

    shutil.copytree(input_dir, output_dir, ignore=ignore_hidden_dirs)
    return output_dir


def _open_video_writer(
    output_mp4: Path, fps: float, width: int, height: int
) -> subprocess.Popen:
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "18",
        "-preset",
        "medium",
        str(output_mp4),
    ]
    return subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _close_video_writer(process: subprocess.Popen, output_mp4: Path) -> None:
    assert process.stdin is not None
    process.stdin.close()
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg failed with return code {return_code}: {output_mp4}")


def process_video(input_mp4: Path, output_mp4: Path) -> None:
    cap = cv2.VideoCapture(str(input_mp4))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {input_mp4}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0:
        fps = 30.0

    aruco_dict = _aruco_dict(ARUCO_DICT_ID)
    parameters = _detector_parameters()
    writer = _open_video_writer(output_mp4, fps, width, height)
    assert writer.stdin is not None

    frame_idx = 0
    detected_count = 0
    last_good_corners = None
    prev_frame = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        raw_frame = frame.copy()

        raw_corners = detect_marker_corners(
            frame, MARKER_ID, aruco_dict, parameters
        )
        corners, last_good_corners = select_corners_with_fallback(
            raw_corners,
            last_good_corners,
            prev_frame,
            frame,
            MAX_MARKER_JUMP_PX,
        )
        if corners is not None:
            frame = remove_marker(frame, corners)
            detected_count += 1

        writer.stdin.write(frame.tobytes())
        prev_frame = raw_frame
        frame_idx += 1

    cap.release()
    _close_video_writer(writer, output_mp4)
    print(f"Saved: {output_mp4}  removed_frames={detected_count}/{frame_idx}")


def run(
    camera_name: str,
    dataset_dir: Path,
) -> Path:
    if not hasattr(cv2, "xphoto") or not hasattr(cv2.xphoto, "inpaint"):
        raise RuntimeError("OpenCV xphoto.inpaint is required for FSR_FAST inpainting.")
    if not camera_name:
        raise ValueError("camera_name must be a non-empty string.")

    input_dir = Path(dataset_dir)
    output_dir = copy_dataset_dir(input_dir)
    print(f"Copied dataset dir to: {output_dir}")

    mp4_paths = list(iter_target_mp4(input_dir, camera_name))
    print(f"Found {camera_name} videos: {len(mp4_paths)}")
    if not mp4_paths:
        raise FileNotFoundError(
            f"No {camera_name}_rgb_image.rmb.mp4 found under {input_dir}"
        )

    for mp4_path in mp4_paths:
        relative_mp4 = mp4_path.relative_to(input_dir)
        process_video(mp4_path, output_dir / relative_mp4)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove one ArUco marker from RMB RGB videos with OpenCV FSR_FAST inpainting."
    )
    parser.add_argument("camera_name", help="Video camera name, e.g. front")
    parser.add_argument("dataset_dir", type=Path, help="Dataset directory")
    args = parser.parse_args()

    run(args.camera_name, args.dataset_dir.expanduser().resolve())


if __name__ == "__main__":
    main()
