import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2]))

from robo_manip_baselines.misc.arucoboard.arucoboard_modules.aruco_prompt import (
    ArucoPromptGenerator,
    expand_corners,
)

MARKER_LENGTH_M = 0.02940
MARKER_SEPARATION_M = 0.0050
MARKERS_X = 5
MARKERS_Y = 7
GRIDBOARD_W_M = MARKERS_X * MARKER_LENGTH_M + (MARKERS_X - 1) * MARKER_SEPARATION_M
GRIDBOARD_H_M = MARKERS_Y * MARKER_LENGTH_M + (MARKERS_Y - 1) * MARKER_SEPARATION_M
MAX_JUMP_PX = 10.0


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


def apply_whiteout(
    frame: np.ndarray,
    corners: np.ndarray,
    box_mask: np.ndarray | None = None,
    scale: int = 4,
) -> np.ndarray:
    height, width = frame.shape[:2]
    corners = expand_corners(corners)
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
    # fill_color = np.array([0x4D, 0x78, 0x9A], dtype=np.float32)

    # fill_color = np.array([0x52, 0x86, 0xA3], dtype=np.float32)  # #a38652 RGB
    fill_color = np.array([0x5A, 0x8C, 0xA7], dtype=np.float32)  # #a78c5a RGB


    frame_f = frame_f * (1.0 - alpha[..., None]) + fill_color * alpha[..., None]
    return np.clip(frame_f, 0.0, 255.0).astype(np.uint8)


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


def run(camera_name: str, dataset_dir: Path, intrinsics_path: Path | None = None) -> None:
    _require_repo_cwd()
    if not camera_name:
        raise ValueError("camera_name must be a non-empty string.")
    input_dir = Path(dataset_dir)
    output_dir = copy_dataset_dir(input_dir)
    print(f"Copied dataset dir to: {output_dir}")

    gen = ArucoPromptGenerator(camera_name=camera_name, intrinsics_path=intrinsics_path)
    mp4_paths = list(iter_target_mp4(input_dir, camera_name))
    print(f"Found {camera_name} videos: {len(mp4_paths)}")

    for mp4_path in mp4_paths:
        mask_dir = mask_dir_for_mp4(mp4_path, camera_name)
        if not mask_dir.exists():
            raise FileNotFoundError(f"Missing mask dir: {mask_dir}")

        relative_mp4 = mp4_path.relative_to(input_dir)
        output_mp4 = output_dir / relative_mp4
        tmp_path = output_mp4.with_suffix(".tmp.mp4")

        cap = cv2.VideoCapture(str(mp4_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {mp4_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if fps <= 0:
            fps = 30.0

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(tmp_path), fourcc, fps, (width, height))

        frame_idx = 0
        last_good_corners = None
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
                box_mask_path = mask_path_for_frame(mask_dir, frame_idx)
                if not box_mask_path.exists():
                    raise FileNotFoundError(f"Missing mask file: {box_mask_path}")
                box_mask = np.load(box_mask_path)
                frame = apply_whiteout(frame, corners, box_mask=box_mask)

            writer.write(frame)
            frame_idx += 1

        cap.release()
        writer.release()

        ffmpeg = "ffmpeg"
        cmd = [
            ffmpeg,
            "-y",
            "-i",
            str(tmp_path),
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
        try:
            subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            print(f"Saved: {output_mp4}")
            tmp_path.unlink(missing_ok=True)
        except FileNotFoundError:
            print("ffmpeg not found; kept intermediate mp4.")
        except subprocess.CalledProcessError as exc:
            print(f"ffmpeg failed: {exc}")


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
