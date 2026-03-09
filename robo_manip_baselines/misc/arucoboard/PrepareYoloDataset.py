from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Tuple

import cv2
import h5py
import numpy as np
import yaml

from arucoboard_modules.prepare_bedrooms import (
    bedroom_dir_for_mp4,
    iter_target_mp4,
    prepare_bedrooms,
)

MAX_JUMP_PX = 50.0


def load_corners(hdf5_path: Path, key: str) -> np.ndarray:
    with h5py.File(hdf5_path, "r") as h5file:
        if key not in h5file:
            raise KeyError(f"Missing dataset key: {key}")
        corners = h5file[key][:]
    return corners


def is_valid_corners(corners: np.ndarray) -> bool:
    if corners.shape != (4, 2):
        return False
    return np.all(corners >= 0)


def _is_jump_outlier(
    corners: np.ndarray, prev_corners: np.ndarray, max_jump_px: float
) -> bool:
    deltas = np.linalg.norm(corners - prev_corners, axis=-1)
    return bool(np.any(deltas > max_jump_px))


def remove_jump_outliers(corners_seq: np.ndarray, max_jump_px: float) -> np.ndarray:
    cleaned = np.asarray(corners_seq).copy()
    last_good = None
    for idx in range(len(cleaned)):
        corners = np.asarray(cleaned[idx])
        if last_good is not None and _is_jump_outlier(corners, last_good, max_jump_px):
            cleaned[idx] = last_good.copy()
        else:
            last_good = corners.copy()
    return cleaned


def _list_frame_paths(bedroom_dir: Path) -> List[Path]:
    frame_paths = [
        p
        for p in bedroom_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg"}
    ]
    frame_paths.sort(key=lambda p: int(p.stem))
    return frame_paths


def _load_image_shape(image_path: Path) -> Tuple[int, int]:
    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    height, width = img.shape[:2]
    return width, height


def _to_yolo_line(
    corners: np.ndarray, width: int, height: int, class_index: int
) -> str:
    corners = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    x_min = float(np.min(corners[:, 0]))
    x_max = float(np.max(corners[:, 0]))
    y_min = float(np.min(corners[:, 1]))
    y_max = float(np.max(corners[:, 1]))

    xc = (x_min + x_max) / 2.0
    yc = (y_min + y_max) / 2.0
    w = x_max - x_min
    h = y_max - y_min

    xc /= width
    yc /= height
    w /= width
    h /= height

    kpts = []
    visibility = 2.0
    for x, y in corners:
        kpts.append(x / width)
        kpts.append(y / height)
        kpts.append(visibility)

    values = [class_index, xc, yc, w, h] + kpts
    return " ".join(f"{v:.6f}" if isinstance(v, float) else str(v) for v in values)


def _write_labels_for_video(
    mp4_path: Path,
    camera_name: str,
    corners_key: str,
    class_index: int,
    max_jump_px: float,
) -> Tuple[Path, Path]:
    hdf5_path = mp4_path.parent / "main.rmb.hdf5"
    if not hdf5_path.exists():
        raise FileNotFoundError(f"Missing HDF5: {hdf5_path}")

    bedroom_dir = bedroom_dir_for_mp4(mp4_path)
    if not bedroom_dir.exists():
        raise FileNotFoundError(f"Missing bedroom dir: {bedroom_dir}")

    frame_paths = _list_frame_paths(bedroom_dir)
    if not frame_paths:
        raise RuntimeError(f"No frames found in: {bedroom_dir}")

    width, height = _load_image_shape(frame_paths[0])

    corners_seq = load_corners(hdf5_path, corners_key)
    corners_seq = remove_jump_outliers(corners_seq, max_jump_px)

    num_frames = min(len(frame_paths), len(corners_seq))
    if len(frame_paths) != len(corners_seq):
        print(
            f"[PrepareYoloDataset] WARNING: frame/corner length mismatch: "
            f"frames={len(frame_paths)} corners={len(corners_seq)} for {mp4_path}"
        )

    for idx in range(num_frames):
        frame_path = frame_paths[idx]
        corners = np.asarray(corners_seq[idx]).reshape(4, 2)
        label_path = frame_path.with_suffix(".txt")
        if not is_valid_corners(corners):
            label_path.write_text("")
            continue
        line = _to_yolo_line(corners, width, height, class_index)
        label_path.write_text(line + "\n")

    return bedroom_dir, bedroom_dir


def _split_videos(
    mp4_paths: List[Path], train_ratio: float, seed: int
) -> Tuple[List[Path], List[Path]]:
    if not mp4_paths:
        raise ValueError("No target mp4 files found.")
    if not (0.0 < train_ratio < 1.0):
        raise ValueError("train_ratio must be between 0 and 1.")
    mp4_paths = list(mp4_paths)
    rng = random.Random(seed)
    rng.shuffle(mp4_paths)
    num_train = int(len(mp4_paths) * train_ratio)
    train_mp4 = mp4_paths[:num_train]
    val_mp4 = mp4_paths[num_train:]
    return train_mp4, val_mp4


def _dump_yaml(
    yaml_path: Path,
    dataset_dir: Path,
    train_image_dirs: List[Path],
    val_image_dirs: List[Path],
) -> None:
    rel_train = [str(p.relative_to(dataset_dir)) for p in train_image_dirs]
    rel_val = [str(p.relative_to(dataset_dir)) for p in val_image_dirs]
    data = {
        "path": str(dataset_dir),
        "train": rel_train,
        "val": rel_val,
        "test": "",
        "kpt_shape": [4, 3],
        "names": {0: "box"},
        "kpt_names": {
            0: [
                "top_left",
                "top_right",
                "bottom_right",
                "bottom_left",
            ]
        },
    }
    yaml_path.write_text(yaml.safe_dump(data, sort_keys=False, default_flow_style=False))


def run(
    dataset_dir: Path,
    camera_name: str,
    train_ratio: float,
    seed: int,
    class_index: int,
    max_jump_px: float,
) -> None:
    dataset_dir = Path(dataset_dir).expanduser().resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"dataset_dir not found: {dataset_dir}")

    print(f"[PrepareYoloDataset] dataset_dir: {dataset_dir}")

    prepare_bedrooms(dataset_dir, camera_name)
    mp4_paths = list(iter_target_mp4(dataset_dir, camera_name))

    train_mp4, val_mp4 = _split_videos(mp4_paths, train_ratio, seed)

    train_image_dirs = []
    val_image_dirs = []

    corners_key = f"{camera_name}_aruco_board_corners"

    for mp4_path in train_mp4:
        img_dir, _label_dir = _write_labels_for_video(
            mp4_path,
            camera_name,
            corners_key,
            class_index,
            max_jump_px,
        )
        train_image_dirs.append(img_dir)

    for mp4_path in val_mp4:
        img_dir, _label_dir = _write_labels_for_video(
            mp4_path,
            camera_name,
            corners_key,
            class_index,
            max_jump_px,
        )
        val_image_dirs.append(img_dir)

    yaml_path = dataset_dir / "yolo_front_dataset.yaml"
    _dump_yaml(yaml_path, dataset_dir, train_image_dirs, val_image_dirs)

    print(f"[PrepareYoloDataset] train videos: {len(train_mp4)}")
    print(f"[PrepareYoloDataset] val videos: {len(val_mp4)}")
    print(f"[PrepareYoloDataset] yaml: {yaml_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare YOLO pose dataset from RoboManipBaselines recordings."
    )
    parser.add_argument("dataset_dir", type=str, help="dataset root directory")
    parser.add_argument("--camera", default="front", type=str)
    parser.add_argument("--train-ratio", default=0.8, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--class-index", default=0, type=int)
    parser.add_argument("--max-jump-px", default=MAX_JUMP_PX, type=float)
    args = parser.parse_args()

    run(
        dataset_dir=Path(args.dataset_dir),
        camera_name=args.camera,
        train_ratio=args.train_ratio,
        seed=args.seed,
        class_index=args.class_index,
        max_jump_px=args.max_jump_px,
    )


if __name__ == "__main__":
    main()
