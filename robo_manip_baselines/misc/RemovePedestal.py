import os
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


TARGET_VIDEO_NAME = "front_rgb_image.rmb.mp4"


# ========= 元コード維持 =========
def iter_target_mp4(input_dir: Path):
    for mp4_path in input_dir.rglob(TARGET_VIDEO_NAME):
        if ".bedrooms" in mp4_path.parts:
            continue
        yield mp4_path


def mask_dir_for_mp4(mp4_path: Path) -> Path:
    return mp4_path.parent / ".masks" / "mask_front_rgb_image"


def mask_path_for_frame(mask_dir: Path, frame_idx: int) -> Path:
    return mask_dir / f"{frame_idx:05d}.npy"


def copy_dataset_dir(input_dir: Path) -> Path:
    output_dir = input_dir.with_name(f"{input_dir.name}_masked")

    if output_dir.exists():
        raise FileExistsError(f"{output_dir} already exists")

    def ignore_unnecessary(dir_path: str, names: list[str]) -> list[str]:
        ignored = []
        for name in names:
            path = Path(dir_path) / name

            # 隠しディレクトリ除外（.masks, .bedrooms）
            if name.startswith(".") and path.is_dir():
                ignored.append(name)
                continue

            # 不要ファイル
            if name.endswith(".hdf5"):
                ignored.append(name)
                continue

        return ignored

    shutil.copytree(input_dir, output_dir, ignore=ignore_unnecessary)
    return output_dir


# ========= コア =========
def apply_mask_black(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 3:
        mask = mask.squeeze()

    mask = mask.astype(bool)

    h, w = frame.shape[:2]
    if mask.shape != (h, w):
        mask = cv2.resize(mask.astype(np.uint8), (w, h)) > 0

    frame[mask] = [0, 0, 0]
    return frame


def process_video(mp4_path: Path, input_dir: Path, output_dir: Path):
    mask_dir = mask_dir_for_mp4(mp4_path)

    # ✅ ここが修正ポイント
    if not mask_dir.exists():
        print(f"[SKIP] No mask: {mp4_path}")
        return

    relative_mp4 = mp4_path.relative_to(input_dir)
    output_mp4 = output_dir / relative_mp4
    tmp_path = output_mp4.with_suffix(".tmp.mp4")

    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {mp4_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(tmp_path), fourcc, fps, (width, height))

    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        mask_path = mask_path_for_frame(mask_dir, frame_idx)

        if not mask_path.exists():
            print(f"[WARN] Missing mask frame {frame_idx}, stop video.")
            break

        mask = np.load(mask_path)

        frame = apply_mask_black(frame, mask)
        writer.write(frame)

        frame_idx += 1

    cap.release()
    writer.release()

    # ffmpegで確定
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(tmp_path),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        "-preset", "medium",
        str(output_mp4),
    ]

    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        os.remove(tmp_path)
        print(f"[OK] Saved: {output_mp4}")
    except Exception as e:
        print(f"[ERROR] ffmpeg failed: {e}")
        print(f"[WARN] tmp kept: {tmp_path}")


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python RemovePedestal.py <input_dir>")

    input_dir = Path(sys.argv[1]).resolve()
    output_dir = copy_dataset_dir(input_dir)

    print(f"Copied dataset dir to: {output_dir}")

    mp4_paths = list(iter_target_mp4(input_dir))
    print(f"Found videos: {len(mp4_paths)}")

    for mp4_path in mp4_paths:
        print(f"Processing: {mp4_path}")
        process_video(mp4_path, input_dir, output_dir)


if __name__ == "__main__":
    main()