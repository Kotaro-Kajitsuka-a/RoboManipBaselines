import os
import sys
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np


# ========= 設定 =========
TARGET_VIDEO_NAME = "front_rgb_image.rmb.mp4"
MASK_DIR_NAME = ".masks/mask_front_rgb_image"


# ========= ユーティリティ =========
def iter_target_mp4(input_dir: Path):
    for p in input_dir.rglob(TARGET_VIDEO_NAME):
        yield p


def mask_dir_for_mp4(mp4_path: Path) -> Path:
    return mp4_path.parent / MASK_DIR_NAME


def mask_path_for_frame(mask_dir: Path, frame_idx: int) -> Path:
    return mask_dir / f"{frame_idx:05d}.npy"


def copy_dataset_dir(input_dir: Path) -> Path:
    output_dir = input_dir.with_name(f"{input_dir.name}_masked")

    if output_dir.exists():
        raise FileExistsError(f"{output_dir} already exists")

    shutil.copytree(input_dir, output_dir)
    return output_dir


# ========= コア処理 =========
def apply_mask_black(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    mask=True の部分を黒塗り
    """
    if mask.ndim == 3:
        mask = mask.squeeze()

    mask = mask.astype(bool)

    # サイズ不一致対策
    h, w = frame.shape[:2]
    if mask.shape != (h, w):
        mask = cv2.resize(mask.astype(np.uint8), (w, h)) > 0

    frame[mask] = [0, 0, 0]
    return frame


def process_video(input_mp4: Path, output_mp4: Path):
    mask_dir = mask_dir_for_mp4(input_mp4)

    if not mask_dir.exists():
        print(f"[SKIP] No mask: {input_mp4}")
        return

    tmp_mp4 = output_mp4.with_suffix(".tmp.mp4")

    cap = cv2.VideoCapture(str(input_mp4))
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {input_mp4}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # OpenCV出力（互換性低いが一旦これでOK）
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(tmp_mp4), fourcc, fps, (width, height))

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        mask_path = mask_path_for_frame(mask_dir, frame_idx)

        if not mask_path.exists():
            print(f"[WARN] Missing mask at frame {frame_idx}, stop.")
            break

        mask = np.load(mask_path)

        frame = apply_mask_black(frame, mask)
        writer.write(frame)

        frame_idx += 1

    cap.release()
    writer.release()

    print(f"[INFO] OpenCV write done: {tmp_mp4}")

    # ========= ここが超重要 =========
    # ffmpegで確実に再エンコード（互換性問題を解消）
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(tmp_mp4),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        "-preset", "medium",
        str(output_mp4),
    ]

    try:
        subprocess.run(cmd, check=True)
        os.remove(tmp_mp4)
        print(f"[OK] Saved: {output_mp4}")
    except Exception as e:
        print(f"[ERROR] ffmpeg failed: {e}")
        print(f"[WARN] tmp file remains: {tmp_mp4}")


# ========= メイン =========
def main():
    if len(sys.argv) < 2:
        print("Usage: python simple_mask_ffmpeg.py <input_dir>")
        sys.exit(1)

    input_dir = Path(sys.argv[1]).resolve()
    output_dir = copy_dataset_dir(input_dir)

    print(f"[INFO] Copied to: {output_dir}")

    mp4_list = list(iter_target_mp4(input_dir))
    print(f"[INFO] Found {len(mp4_list)} videos")

    for mp4_path in mp4_list:
        rel = mp4_path.relative_to(input_dir)
        out_mp4 = output_dir / rel

        print(f"\n[PROCESS] {mp4_path}")
        process_video(mp4_path, out_mp4)


if __name__ == "__main__":
    main()