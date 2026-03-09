import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import cv2
import h5py
import numpy as np

MARKER_LENGTH_M = 0.02940
MARKER_SEPARATION_M = 0.0050
MARKERS_X = 5
MARKERS_Y = 7
GRIDBOARD_W_M = MARKERS_X * MARKER_LENGTH_M + (MARKERS_X - 1) * MARKER_SEPARATION_M
GRIDBOARD_H_M = MARKERS_Y * MARKER_LENGTH_M + (MARKERS_Y - 1) * MARKER_SEPARATION_M
WIDTH_MARGIN_M = 0.0125 # actual 0.012
HEIGHT_MARGIN_M = 0.0425 #actual 0.022
MAX_JUMP_PX = 50.0


def load_corners(hdf5_path: str, key: str) -> np.ndarray:
    with h5py.File(hdf5_path, "r") as h5file:
        if key not in h5file:
            raise KeyError(f"Missing dataset key: {key}")
        corners = h5file[key][:]
    return corners


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


def _expand_corners(corners: np.ndarray) -> np.ndarray:
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

    top_height_px = (HEIGHT_MARGIN_M * top_len) / GRIDBOARD_H_M
    bottom_height_px = (HEIGHT_MARGIN_M * bottom_len) / GRIDBOARD_H_M
    left_width_px = (WIDTH_MARGIN_M * left_len) / GRIDBOARD_W_M
    right_width_px = (WIDTH_MARGIN_M * right_len) / GRIDBOARD_W_M

    p0e = p0 - top_unit * top_height_px - left_unit * left_width_px
    p1e = p1 + top_unit * top_height_px - right_unit * right_width_px
    p2e = p2 + bottom_unit * bottom_height_px + right_unit * right_width_px
    p3e = p3 - bottom_unit * bottom_height_px + left_unit * left_width_px
    return np.stack([p0e, p1e, p2e, p3e], axis=0)


def _is_jump_outlier(
    corners: np.ndarray, prev_corners: np.ndarray, max_jump_px: float
) -> bool:
    
    # print(f"corners: {corners}")
    # print(f"prev_corners: {prev_corners}")
    # print(f"diff{corners - prev_corners}")
    deltas = np.linalg.norm(corners - prev_corners, axis=-1) #shape 
    
    return bool(np.any(deltas > max_jump_px))


def remove_jump_outliers(corners_seq: np.ndarray, max_jump_px: float) -> np.ndarray: 
    cleaned = np.asarray(corners_seq).copy()
    last_good: Optional[np.ndarray] = None

    #print(f"corners_seq: {corners_seq[:50]}")

    for idx in range(len(cleaned)):
        corners = np.asarray(cleaned[idx])
        if last_good is not None and _is_jump_outlier(corners, last_good, max_jump_px):
            print(f"\n\nframe {idx}, there is a jump \n\n")
            cleaned[idx] = last_good.copy()
            
        else:
            last_good = corners.copy()



    return cleaned


def apply_whiteout(
    frame: np.ndarray,
    corners: np.ndarray,
    box_mask: Optional[np.ndarray] = None,
    scale: int = 4, # maskを一度scale倍して高解像度にする。maskがギザギザになるのを防ぐ。
) -> np.ndarray:
    height, width = frame.shape[:2]
    corners = _expand_corners(corners)
    scale = max(2, int(scale)) 
    # マスクを高解像度で描いて縮小し、マスクの直線部分がギザギザになるのを防ぐ
    mask = np.zeros((height * scale, width * scale), dtype=np.uint8)
    corners_scaled = (corners * scale).astype(np.int32).reshape(1, 4, 2)
    cv2.fillPoly(mask, [corners_scaled], color=255)
    alpha = cv2.resize(mask, (width, height), interpolation=cv2.INTER_AREA).astype(
        np.float32
    )
    # 塗りつぶしの際、内側はほぼ1、外側は0で、境界だけ中間値になるアルファ
    alpha = np.clip(alpha / 255.0, 0.0, 1.0)
    if box_mask is not None:
        box_mask = _normalize_mask(box_mask, height, width)
        alpha = alpha * box_mask.astype(np.float32)
    frame_f = frame.astype(np.float32)
    # fill_color = np.array([0xB1, 0xB3, 0xBA], dtype=np.float32) #white
    #fill_color = np.array([0x65, 0x79, 0x8F], dtype=np.float32)  # brown #8f7965
    fill_color = np.array([0x4D, 0x78, 0x9A], dtype=np.float32)  # 貼り付け用brown #9a784d (BGR)


    frame_f = frame_f * (1.0 - alpha[..., None]) + fill_color * alpha[..., None]
    return np.clip(frame_f, 0.0, 255.0).astype(np.uint8)


def iter_target_mp4(input_dir: Path):
    target_name = "front_rgb_image.rmb.mp4"
    for mp4_path in input_dir.rglob(target_name):
        if ".bedrooms" in mp4_path.parts:
            continue
        yield mp4_path


def hdf5_for_mp4(mp4_path: Path) -> Path:
    return mp4_path.parent / "main.rmb.hdf5"


def mask_dir_for_mp4(mp4_path: Path) -> Path:
    return mp4_path.parent / ".masks" / "mask_front_rgb_image"


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

    shutil.copytree(input_dir, output_dir, ignore=ignore_hidden_dirs)
    return output_dir


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python RemoveArucoBoard.py <input_dir>")

    input_dir = Path(sys.argv[1]).expanduser().resolve()
    output_dir = copy_dataset_dir(input_dir)
    print(f"Copied dataset dir to: {output_dir}")

    corners_key = "front_aruco_board_corners"
    mp4_paths = list(iter_target_mp4(input_dir))
    print(f"Found front videos: {len(mp4_paths)}")

    for mp4_path in mp4_paths:
        # if not "W_2nd_dir/RealXarm7DualDemo_world0_002.rmb" in str(mp4_path) :
        #     continue

        hdf5_path = hdf5_for_mp4(mp4_path)
        if not hdf5_path.exists():
            raise FileNotFoundError(f"Missing HDF5: {hdf5_path}")
        mask_dir = mask_dir_for_mp4(mp4_path)
        if not mask_dir.exists():
            raise FileNotFoundError(f"Missing mask dir: {mask_dir}")

        relative_mp4 = mp4_path.relative_to(input_dir)
        output_mp4 = output_dir / relative_mp4
        tmp_path = output_mp4.with_suffix(".tmp.mp4")

        corners_seq = load_corners(str(hdf5_path), corners_key)
        corners_seq = remove_jump_outliers(corners_seq, MAX_JUMP_PX)

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
        max_frames = min(
            len(corners_seq),
            int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) or len(corners_seq),
        )

        while frame_idx < max_frames:
            ok, frame = cap.read()
            if not ok:
                break

            corners = np.asarray(corners_seq[frame_idx]).reshape(4, 2)
            box_mask_path = mask_path_for_frame(mask_dir, frame_idx)
            if not box_mask_path.exists():
                raise FileNotFoundError(f"Missing mask file: {box_mask_path}")
            box_mask = np.load(box_mask_path)
            if is_valid_corners(corners):
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
            os.remove(tmp_path)
        except FileNotFoundError:
            print("ffmpeg not found; kept intermediate mp4.")
        except subprocess.CalledProcessError as exc:
            print(f"ffmpeg failed: {exc}")


if __name__ == "__main__":
    main()
