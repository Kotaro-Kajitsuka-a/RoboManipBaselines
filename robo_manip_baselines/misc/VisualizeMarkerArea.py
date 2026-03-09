import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[2]))

from robo_manip_baselines.misc.arucoboard.arucoboard_modules.aruco_prompt import (
    ArucoPromptGenerator,
    expand_corners,
)

MAX_JUMP_PX = 50.0


def is_valid_corners(corners: np.ndarray) -> bool:
    if corners.shape != (4, 2):
        return False
    return np.all(corners >= 0)


def _is_jump_outlier(
    corners: np.ndarray, prev_corners: np.ndarray, max_jump_px: float
) -> bool:
    deltas = np.linalg.norm(corners - prev_corners, axis=-1)
    return bool(np.any(deltas > max_jump_px))


def build_marker_mask(frame: np.ndarray, corners: np.ndarray, scale: int = 4) -> np.ndarray:
    height, width = frame.shape[:2]
    corners = expand_corners(corners)
    scale = max(2, int(scale))

    highres_mask = np.zeros((height * scale, width * scale), dtype=np.uint8)
    corners_scaled = (corners * scale).astype(np.int32).reshape(1, 4, 2)
    cv2.fillPoly(highres_mask, [corners_scaled], color=255)

    mask = cv2.resize(highres_mask, (width, height), interpolation=cv2.INTER_AREA)
    return mask >= 127


def save_mask_png(mask_bool: np.ndarray, output_dir: Path, frame_idx: int) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_name = f"{frame_idx:05d}"
    png_path = output_dir / f"{frame_name}.png"
    Image.fromarray(mask_bool.astype(np.uint8) * 255).save(png_path)
    return png_path


def run(
    camera_name: str,
    input_mp4: Path,
    output_dir: Path,
    intrinsics_path: Path | None = None,
) -> None:
    if not input_mp4.exists():
        raise FileNotFoundError(f"input_mp4 not found: {input_mp4}")
    if not input_mp4.is_file():
        raise ValueError(f"input_mp4 is not a file: {input_mp4}")

    cap = cv2.VideoCapture(str(input_mp4))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {input_mp4}")

    gen = ArucoPromptGenerator(camera_name=camera_name, intrinsics_path=intrinsics_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_idx = 0
    last_good_corners = None
    saved_count = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        _ = gen.infer(frame)
        corners = gen.last_corners()

        if corners is not None:
            corners = np.asarray(corners, dtype=np.float32).reshape(4, 2)
            if last_good_corners is not None and _is_jump_outlier(
                corners, last_good_corners, MAX_JUMP_PX
            ):
                corners = last_good_corners.copy()
            else:
                last_good_corners = corners.copy()

        if corners is not None and is_valid_corners(corners):
            marker_mask = build_marker_mask(frame, corners)
        else:
            marker_mask = np.zeros(frame.shape[:2], dtype=bool)

        save_mask_png(marker_mask, output_dir, frame_idx)
        saved_count += 1
        frame_idx += 1

    cap.release()

    if saved_count == 0:
        raise RuntimeError(f"No frames found in video: {input_mp4}")

    print(f"Saved marker area pngs to: {output_dir}")
    print(f"saved frames: {saved_count}")


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit(
            "Usage: python VisualizeMarkerArea.py <camera_name> <input_mp4> [output_dir]"
        )

    camera_name = sys.argv[1]
    input_mp4 = Path(sys.argv[2]).expanduser().resolve()

    if len(sys.argv) >= 4:
        output_dir = Path(sys.argv[3]).expanduser().resolve()
    else:
        output_dir = input_mp4.parent / ".markerareas"

    run(camera_name=camera_name, input_mp4=input_mp4, output_dir=output_dir)


if __name__ == "__main__":
    main()
