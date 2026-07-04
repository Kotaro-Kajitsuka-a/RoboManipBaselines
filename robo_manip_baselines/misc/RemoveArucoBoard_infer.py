import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from cv2 import aruco

sys.path.append(str(Path(__file__).resolve().parents[2]))

from robo_manip_baselines.misc.arucoboard.arucoboard_modules.aruco_prompt import (
    ArucoPromptGenerator,
)


@dataclass(frozen=True)
class BoardRemovalConfig:
    board_type: str
    inpainting_color: str
    margin_m: tuple[float, float]


@dataclass(frozen=True)
class BoardRuntime:
    config: BoardRemovalConfig
    board_w_m: float
    board_h_m: float
    scales: tuple[float, ...]
    max_jump_px: float
    aruco_dict: object | None = None
    parameters: object | None = None
    board: object | None = None


# ========= User settings =========
# board_type: "big", "small", or "long"
# inpainting_color: "#RRGGBB"
# margin_m: (width_margin_m, height_margin_m)
BOARD_REMOVAL_CONFIGS = [
    BoardRemovalConfig(
        board_type="big",
        inpainting_color="#c8a283",
        margin_m=(0.0220, 0.0330),
    ),
    BoardRemovalConfig(
        board_type="small",
        inpainting_color="#4A2F12",
        margin_m=(0.038, 0.0025),
    ),
    # BoardRemovalConfig(
    #     board_type="long",
    #     inpainting_color="#FBB88D",
    #     margin_m=(0.0100, 0.0100),
    # ),
]

MARKER_LENGTH_M = 0.02940
MARKER_SEPARATION_M = 0.0050
MARKERS_X = 5
MARKERS_Y = 7
GRIDBOARD_W_M = MARKERS_X * MARKER_LENGTH_M + (MARKERS_X - 1) * MARKER_SEPARATION_M
GRIDBOARD_H_M = MARKERS_Y * MARKER_LENGTH_M + (MARKERS_Y - 1) * MARKER_SEPARATION_M
BIG_BOARD_MAX_JUMP_PX = 10.0
SMALL_BOARD_MAX_JUMP_PX = 60.0
LONG_BOARD_MAX_JUMP_PX = 10.0

SMALL_BOARD_DICT_ID = aruco.DICT_5X5_250
SMALL_BOARD_MARKER_IDS = [100, 101, 102]
SMALL_BOARD_MARKER_LENGTH_M = 0.0287
SMALL_BOARD_MARKER_GAP_M = 0.005
SMALL_BOARD_W_M = SMALL_BOARD_MARKER_LENGTH_M * len(
    SMALL_BOARD_MARKER_IDS
) + SMALL_BOARD_MARKER_GAP_M * (len(SMALL_BOARD_MARKER_IDS) - 1)
SMALL_BOARD_H_M = SMALL_BOARD_MARKER_LENGTH_M

# Dummy values for a future marker board. Replace these when the long marker is fixed.
LONG_BOARD_W_M = 0.1000
LONG_BOARD_H_M = 0.0300


def rgb_hex_to_bgr(rgb_hex: str) -> tuple[int, int, int]:
    if not rgb_hex.startswith("#") or len(rgb_hex) != 7:
        raise ValueError(f"Expected '#RRGGBB', got: {rgb_hex}")
    red = int(rgb_hex[1:3], 16)
    green = int(rgb_hex[3:5], 16)
    blue = int(rgb_hex[5:7], 16)
    return blue, green, red


def is_valid_corners(corners: np.ndarray) -> bool:
    if corners.shape != (4, 2):
        return False
    return np.all(corners >= 0)


def _normalize_mask(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    mask = np.asarray(mask)
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]
    if mask.shape != (height, width):
        raise ValueError(f"Mask shape mismatch: {mask.shape} != ({height}, {width})")
    return mask.astype(bool)


def _is_jump_outlier(
    corners: np.ndarray, prev_corners: np.ndarray, max_jump_px: float
) -> bool:
    deltas = np.linalg.norm(corners - prev_corners, axis=-1)
    return bool(np.any(deltas > max_jump_px))


def _is_raw_corners_outlier(
    corners: np.ndarray,
    prev_corners: np.ndarray | None,
    max_jump_px: float,
) -> bool:
    if prev_corners is None:
        return False
    return _is_jump_outlier(corners, prev_corners, max_jump_px)


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
    _, otsu = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    base_images = (gray, contrast, sharpened, otsu)
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


def detect_markers_robust(
    image: np.ndarray,
    aruco_dict,
    parameters,
    scales: tuple[float, ...],
):
    for candidate, scale in _candidate_gray_images(image, scales):
        corners, ids, _ = aruco.detectMarkers(
            candidate, aruco_dict, parameters=parameters
        )
        if ids is None or len(ids) == 0:
            continue
        if scale != 1.0:
            corners = [corner / scale for corner in corners]
        return corners, ids
    return None, None


def expand_corners(
    corners: np.ndarray,
    board_w_m: float,
    board_h_m: float,
    width_margin_m: float,
    height_margin_m: float,
) -> np.ndarray:
    corners = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    p0, p1, p2, p3 = corners
    top_vec = p1 - p0
    bottom_vec = p2 - p3
    left_vec = p3 - p0
    right_vec = p2 - p1
    top_len = float(np.linalg.norm(top_vec))
    bottom_len = float(np.linalg.norm(bottom_vec))
    left_len = float(np.linalg.norm(left_vec))
    right_len = float(np.linalg.norm(right_vec))
    if min(top_len, bottom_len, left_len, right_len) < 1e-6:
        return corners

    top_unit = top_vec / top_len
    bottom_unit = bottom_vec / bottom_len
    left_unit = left_vec / left_len
    right_unit = right_vec / right_len

    top_height_px = (height_margin_m * top_len) / board_h_m
    bottom_height_px = (height_margin_m * bottom_len) / board_h_m
    left_width_px = (width_margin_m * left_len) / board_w_m
    right_width_px = (width_margin_m * right_len) / board_w_m

    p0e = p0 - top_unit * top_height_px - left_unit * left_width_px
    p1e = p1 + top_unit * top_height_px - right_unit * right_width_px
    p2e = p2 + bottom_unit * bottom_height_px + right_unit * right_width_px
    p3e = p3 - bottom_unit * bottom_height_px + left_unit * left_width_px
    return np.stack([p0e, p1e, p2e, p3e], axis=0)


def _build_big_runtime(config: BoardRemovalConfig) -> BoardRuntime:
    aruco_dict = _aruco_dict(aruco.DICT_4X4_50)
    parameters = _detector_parameters()
    board = aruco.GridBoard(
        (MARKERS_X, MARKERS_Y), MARKER_LENGTH_M, MARKER_SEPARATION_M, aruco_dict
    )
    return BoardRuntime(
        config=config,
        board_w_m=GRIDBOARD_W_M,
        board_h_m=GRIDBOARD_H_M,
        scales=(1.0, 1.5),
        max_jump_px=BIG_BOARD_MAX_JUMP_PX,
        aruco_dict=aruco_dict,
        parameters=parameters,
        board=board,
    )


def _build_small_runtime(config: BoardRemovalConfig) -> BoardRuntime:
    aruco_dict = _aruco_dict(SMALL_BOARD_DICT_ID)
    parameters = _detector_parameters()
    board = aruco.GridBoard(
        (len(SMALL_BOARD_MARKER_IDS), 1),
        SMALL_BOARD_MARKER_LENGTH_M,
        SMALL_BOARD_MARKER_GAP_M,
        aruco_dict,
        np.array(SMALL_BOARD_MARKER_IDS, dtype=np.int32),
    )
    return BoardRuntime(
        config=config,
        board_w_m=SMALL_BOARD_W_M,
        board_h_m=SMALL_BOARD_H_M,
        scales=(1.0, 1.5, 2.0),
        max_jump_px=SMALL_BOARD_MAX_JUMP_PX,
        aruco_dict=aruco_dict,
        parameters=parameters,
        board=board,
    )


def _build_long_runtime(config: BoardRemovalConfig) -> BoardRuntime:
    return BoardRuntime(
        config=config,
        board_w_m=LONG_BOARD_W_M,
        board_h_m=LONG_BOARD_H_M,
        scales=(1.0, 1.5),
        max_jump_px=LONG_BOARD_MAX_JUMP_PX,
    )


def build_board_runtime(config: BoardRemovalConfig) -> BoardRuntime:
    if config.board_type == "big":
        return _build_big_runtime(config)
    if config.board_type == "small":
        return _build_small_runtime(config)
    if config.board_type == "long":
        return _build_long_runtime(config)
    raise ValueError(f"Unknown board_type: {config.board_type}")


def detect_board_corners(
    image: np.ndarray,
    K: np.ndarray,
    dist_coeffs: np.ndarray,
    aruco_dict,
    parameters,
    board,
    board_w_m: float,
    board_h_m: float,
    scales: tuple[float, ...],
) -> np.ndarray | None:
    corners, ids = detect_markers_robust(image, aruco_dict, parameters, scales)
    if ids is None or len(ids) == 0:
        return None

    retval, rvec, tvec = aruco.estimatePoseBoard(
        corners, ids, board, K, dist_coeffs, None, None
    )
    if retval <= 0:
        return None

    board_corners_3d = np.array(
        [
            [0.0, 0.0, 0.0],
            [board_w_m, 0.0, 0.0],
            [board_w_m, board_h_m, 0.0],
            [0.0, board_h_m, 0.0],
        ],
        dtype=np.float32,
    )
    proj, _ = cv2.projectPoints(board_corners_3d, rvec, tvec, K, dist_coeffs)
    return proj.reshape(4, 2).astype(np.float32)


def detect_long_board_corners(
    image: np.ndarray,
    K: np.ndarray,
    dist_coeffs: np.ndarray,
) -> np.ndarray | None:
    # TODO: Implement when the long marker layout is fixed.
    return None


def detect_runtime_corners(
    image: np.ndarray,
    K: np.ndarray,
    dist_coeffs: np.ndarray,
    runtime: BoardRuntime,
) -> np.ndarray | None:
    if runtime.config.board_type == "long":
        return detect_long_board_corners(image, K, dist_coeffs)

    assert runtime.aruco_dict is not None
    assert runtime.parameters is not None
    assert runtime.board is not None
    return detect_board_corners(
        image,
        K,
        dist_coeffs,
        runtime.aruco_dict,
        runtime.parameters,
        runtime.board,
        runtime.board_w_m,
        runtime.board_h_m,
        runtime.scales,
    )


def track_corners(
    prev_frame: np.ndarray,
    frame: np.ndarray,
    prev_corners: np.ndarray,
    max_point_move_px: float = 80.0,
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
        if not _is_raw_corners_outlier(raw_corners, last_good_corners, max_jump_px):
            return raw_corners, raw_corners.copy()

    if prev_frame is not None and last_good_corners is not None:
        tracked_corners = track_corners(prev_frame, frame, last_good_corners)
        if tracked_corners is not None:
            return tracked_corners, tracked_corners.copy()

    if last_good_corners is not None:
        return last_good_corners.copy(), last_good_corners
    return None, last_good_corners


def apply_whiteout(
    frame: np.ndarray,
    corners: np.ndarray,
    board_w_m: float,
    board_h_m: float,
    width_margin_m: float,
    height_margin_m: float,
    fill_color_bgr: tuple[int, int, int],
    box_mask: np.ndarray | None = None,
    scale: int = 4,
) -> np.ndarray:
    height, width = frame.shape[:2]
    corners = expand_corners(
        corners, board_w_m, board_h_m, width_margin_m, height_margin_m
    )
    scale = max(2, int(scale))
    mask = np.zeros((height * scale, width * scale), dtype=np.uint8)
    corners_scaled = (corners * scale).astype(np.int32).reshape(1, 4, 2)
    cv2.fillPoly(mask, [corners_scaled], color=255)
    alpha = cv2.resize(mask, (width, height), interpolation=cv2.INTER_AREA).astype(
        np.float32
    )
    alpha = np.clip(alpha / 255.0, 0.0, 1.0)
    if box_mask is not None:
        box_mask = _normalize_mask(box_mask, height, width)
        alpha = alpha * box_mask.astype(np.float32)
    frame_f = frame.astype(np.float32)
    fill_color = np.array(fill_color_bgr, dtype=np.float32)

    frame_f = frame_f * (1.0 - alpha[..., None]) + fill_color * alpha[..., None]
    return np.clip(frame_f, 0.0, 255.0).astype(np.uint8)


def board_removal_spec(runtime: BoardRuntime, corners: np.ndarray):
    width_margin_m, height_margin_m = runtime.config.margin_m
    return (
        corners,
        runtime.board_w_m,
        runtime.board_h_m,
        width_margin_m,
        height_margin_m,
        rgb_hex_to_bgr(runtime.config.inpainting_color),
    )


def iter_target_mp4(input_dir: Path, camera_name: str):
    target_name = f"{camera_name}_rgb_image.rmb.mp4"
    for mp4_path in input_dir.rglob(target_name):
        if ".bedrooms" in mp4_path.parts:
            continue
        yield mp4_path


def mask_dir_for_mp4(mp4_path: Path, camera_name: str) -> Path:
    return mp4_path.parent / ".masks" / f"mask_{camera_name}_rgb_image"


def mask_path_for_frame(mask_dir: Path, frame_idx: int) -> Path:
    return mask_dir / f"{frame_idx:05d}.npy"


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


def copy_dataset_dir(input_dir: Path) -> Path:
    if not input_dir.exists():
        raise FileNotFoundError(f"input_dir not found: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"input_dir is not a directory: {input_dir}")

    output_dir = input_dir.with_name(f"{input_dir.name}_board_removed")
    if output_dir.exists():
        raise FileExistsError(f"output_dir already exists: {output_dir}")

    def ignore_hidden_dirs(dir_path: str, entries: list[str]) -> list[str]:
        ignored: list[str] = []
        for name in entries:
            if name.startswith("."):
                entry_path = Path(dir_path) / name
                if entry_path.is_dir():
                    ignored.append(name)
        return ignored

    import shutil

    shutil.copytree(input_dir, output_dir, ignore=ignore_hidden_dirs)
    return output_dir


def _require_repo_cwd() -> Path:
    cwd = Path(".").resolve()
    if cwd.name != "robo_manip_baselines":
        raise RuntimeError("Run from the 'robo_manip_baselines' directory.")
    return cwd


def run(
    camera_name: str, dataset_dir: Path, intrinsics_path: Path | None = None
) -> None:
    _require_repo_cwd()
    if not camera_name:
        raise ValueError("camera_name must be a non-empty string.")
    input_dir = Path(dataset_dir)
    output_dir = copy_dataset_dir(input_dir)
    print(f"Copied dataset dir to: {output_dir}")

    gen = ArucoPromptGenerator(camera_name=camera_name, intrinsics_path=intrinsics_path)
    runtimes = [build_board_runtime(config) for config in BOARD_REMOVAL_CONFIGS]
    mp4_paths = list(iter_target_mp4(input_dir, camera_name))
    print(f"Found {camera_name} videos: {len(mp4_paths)}")

    for mp4_path in mp4_paths:
        mask_dir = mask_dir_for_mp4(mp4_path, camera_name)
        if not mask_dir.exists():
            raise FileNotFoundError(f"Missing mask dir: {mask_dir}")

        relative_mp4 = mp4_path.relative_to(input_dir)
        output_mp4 = output_dir / relative_mp4

        cap = cv2.VideoCapture(str(mp4_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {mp4_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if fps <= 0:
            fps = 30.0

        writer = _open_video_writer(output_mp4, fps, width, height)
        assert writer.stdin is not None

        frame_idx = 0
        last_good_corners: dict[str, np.ndarray | None] = {
            runtime.config.board_type: None for runtime in runtimes
        }
        prev_frame = None
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            raw_frame = frame.copy()

            removal_specs = []
            for runtime in runtimes:
                board_type = runtime.config.board_type
                raw_corners = detect_runtime_corners(
                    frame, gen.K, gen.dist_coeffs, runtime
                )
                corners, last_good_corners[board_type] = select_corners_with_fallback(
                    raw_corners,
                    last_good_corners[board_type],
                    prev_frame,
                    frame,
                    runtime.max_jump_px,
                )
                if corners is not None and is_valid_corners(corners):
                    removal_specs.append(board_removal_spec(runtime, corners))

            if removal_specs:
                box_mask_path = mask_path_for_frame(mask_dir, frame_idx)
                if not box_mask_path.exists():
                    raise FileNotFoundError(f"Missing mask file: {box_mask_path}")
                box_mask = np.load(box_mask_path)
                for spec in removal_specs:
                    (
                        detected,
                        board_w_m,
                        board_h_m,
                        width_margin_m,
                        height_margin_m,
                        inpainting_color_bgr,
                    ) = spec
                    frame = apply_whiteout(
                        frame,
                        detected,
                        board_w_m,
                        board_h_m,
                        width_margin_m,
                        height_margin_m,
                        inpainting_color_bgr,
                        box_mask=box_mask,
                    )

            writer.stdin.write(frame.tobytes())
            prev_frame = raw_frame
            frame_idx += 1

        cap.release()
        _close_video_writer(writer, output_mp4)
        print(f"Saved: {output_mp4}")


def main() -> None:
    _require_repo_cwd()
    if len(sys.argv) < 3:
        raise SystemExit(
            "Usage: python RemoveArucoBoard_infer.py <camera_name> <dataset_dir>"
        )

    camera_name = sys.argv[1]
    input_dir = Path(sys.argv[2]).expanduser().resolve()
    run(camera_name=camera_name, dataset_dir=input_dir)


if __name__ == "__main__":
    main()
