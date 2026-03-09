import argparse
from pathlib import Path

import cv2
import h5py
import numpy as np

# Fixed output fps for timeline quantization.
# With 60 fps, frame-duration quantization error is up to ~16.7 ms.
TIMELINE_FPS = 60.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create real-time mp4 using per-frame timing in main.rmb.hdf5."
    )
    parser.add_argument("input_mp4", type=str, help="Path to input mp4")
    return parser.parse_args()


def _make_output_path(input_mp4: Path) -> Path:
    name = input_mp4.name
    if name.endswith(".rmb.mp4"):
        return input_mp4.with_name(name[: -len(".rmb.mp4")] + "_realtime.rmb.mp4")
    return input_mp4.with_name(input_mp4.stem + "_realtime.mp4")


def _load_time_sequence(hdf5_path: Path) -> np.ndarray:
    with h5py.File(hdf5_path, "r") as h5file:
        if "time" not in h5file:
            raise KeyError(f"`time` dataset not found in: {hdf5_path}")
        time_seq = np.asarray(h5file["time"][:], dtype=np.float64)
    if time_seq.ndim != 1:
        raise ValueError(f"`time` must be 1D, got shape={time_seq.shape}")
    if len(time_seq) == 0:
        raise ValueError(f"`time` is empty: {hdf5_path}")
    return time_seq


def _build_repeat_counts_from_time(time_seq: np.ndarray) -> np.ndarray:
    t_rel = np.asarray(time_seq, dtype=np.float64) - float(time_seq[0])
    t_rel = np.maximum.accumulate(t_rel)

    if len(t_rel) == 1:
        return np.array([1], dtype=np.int64)

    duration = float(t_rel[-1])
    if duration <= 0.0:
        return np.ones(len(t_rel), dtype=np.int64)

    num_output_frames = int(np.round(duration * TIMELINE_FPS)) + 1
    output_t = np.arange(num_output_frames, dtype=np.float64) / TIMELINE_FPS
    src_idx = np.searchsorted(t_rel, output_t, side="right") - 1
    src_idx = np.clip(src_idx, 0, len(t_rel) - 1)

    repeat_counts = np.bincount(src_idx, minlength=len(t_rel)).astype(np.int64)
    # Quantizationで0回になるフレームを防ぐ
    repeat_counts = np.maximum(repeat_counts, 1)
    return repeat_counts


def _open_cv2_writer(output_mp4: Path, width: int, height: int) -> cv2.VideoWriter:
    for codec in ("mp4v", "avc1"):
        writer = cv2.VideoWriter(
            str(output_mp4),
            cv2.VideoWriter_fourcc(*codec),
            TIMELINE_FPS,
            (width, height),
        )
        if writer.isOpened():
            return writer
    raise RuntimeError(f"Failed to create video writer: {output_mp4}")


def main() -> None:
    args = parse_args()

    input_mp4 = Path(args.input_mp4).expanduser().resolve()
    if not input_mp4.exists():
        raise FileNotFoundError(f"input mp4 not found: {input_mp4}")
    if not input_mp4.is_file():
        raise ValueError(f"input mp4 is not a file: {input_mp4}")
    if input_mp4.name.endswith("_depth_image.rmb.mp4"):
        raise ValueError(
            "Depth video is not supported by this script. Pass an rgb mp4 file."
        )

    hdf5_path = input_mp4.parent / "main.rmb.hdf5"
    if not hdf5_path.exists():
        raise FileNotFoundError(f"main.rmb.hdf5 not found: {hdf5_path}")
    time_seq = _load_time_sequence(hdf5_path)

    cap = cv2.VideoCapture(str(input_mp4))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {input_mp4}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    input_fps = float(cap.get(cv2.CAP_PROP_FPS))
    if input_fps <= 0.0:
        input_fps = 25.0

    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError(f"Invalid video size: {input_mp4}")

    if frame_count > 0:
        assert frame_count == len(time_seq), (
            f"frame_count != len(time): {frame_count} != {len(time_seq)} "
            f"(video={input_mp4}, hdf5={hdf5_path})"
        )

    repeat_counts = _build_repeat_counts_from_time(time_seq)
    output_mp4 = _make_output_path(input_mp4)
    if output_mp4.exists():
        print(f"[CreateRealtimeMp4FromRmbTime] Overwrite existing file: {output_mp4}")
        output_mp4.unlink()

    writer = _open_cv2_writer(output_mp4, width, height)

    written_frames = 0
    try:
        for repeat in repeat_counts:
            ok, bgr = cap.read()
            if not ok:
                raise RuntimeError(
                    f"Video ended early at frame {written_frames} in {input_mp4}"
                )
            for _ in range(int(repeat)):
                writer.write(bgr)
                written_frames += 1

        extra_ok, _ = cap.read()
        if extra_ok:
            raise RuntimeError(
                "Video has more frames than expected from `time` length. "
                f"video={input_mp4}, len(time)={len(time_seq)}"
            )
    finally:
        cap.release()
        writer.release()

    input_duration = float(len(time_seq)) / input_fps
    real_duration = float(time_seq[-1] - time_seq[0]) if len(time_seq) >= 2 else 0.0
    output_duration = float(written_frames) / TIMELINE_FPS
    duration_error = output_duration - real_duration
    duration_error_ms = duration_error * 1e3
    if real_duration > 0.0:
        duration_error_pct = 100.0 * duration_error / real_duration
    else:
        duration_error_pct = 0.0

    print(f"input video : {input_mp4}")
    print(f"input hdf5  : {hdf5_path}")
    print(f"output video: {output_mp4}")
    print(f"fps         : input={input_fps:.6f}, output={TIMELINE_FPS:.6f}")
    print(
        f"durations[s]: input={input_duration:.6f}, real(time)={real_duration:.6f}, output={output_duration:.6f}"
    )
    print(f"frames      : input={len(time_seq)}, output={written_frames}")
    print(
        f"duration err: output-real={duration_error:+.6f} s ({duration_error_ms:+.3f} ms, {duration_error_pct:+.4f} %)"
    )


if __name__ == "__main__":
    main()
