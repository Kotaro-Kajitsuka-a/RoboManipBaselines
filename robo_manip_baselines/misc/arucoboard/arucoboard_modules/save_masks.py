from pathlib import Path

import numpy as np
from PIL import Image


def mask_dir_from_video_dir(video_dir: Path) -> Path:
    video_dir = Path(video_dir)
    if video_dir.name.startswith("bedroom_"):
        mask_name = "mask_" + video_dir.name[len("bedroom_") :]
    else:
        mask_name = "mask_" + video_dir.name
    return video_dir.parent.parent / ".masks" / mask_name


def _save_mask(mask_bool: np.ndarray, mask_dir: Path, frame_name: str):
    mask_dir = Path(mask_dir)
    mask_dir.mkdir(parents=True, exist_ok=True)

    mask_u8 = (mask_bool.astype(np.uint8) * 255)
    if mask_u8.ndim == 3 and mask_u8.shape[0] == 1:
        mask_u8 = mask_u8[0]
    png_path = mask_dir / f"{frame_name}.png"
    npy_path = mask_dir / f"{frame_name}.npy"

    Image.fromarray(mask_u8).save(png_path)
    np.save(npy_path, mask_bool)
    return png_path, npy_path


def save_all_masks(mask_bools, mask_dir: Path):
    mask_dir = Path(mask_dir)
    mask_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for i, mask_bool in enumerate(mask_bools):
        frame_name = f"{i:05d}"
        saved.append(_save_mask(mask_bool, mask_dir, frame_name))
    return saved
