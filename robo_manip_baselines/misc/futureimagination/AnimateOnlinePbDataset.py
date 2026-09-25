"""Animate one RMB episode, frame-aligned with its recorded camera videos."""

import argparse
import json
import subprocess
from fractions import Fraction
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter

from robo_manip_baselines.misc.futureimagination.PlotOnlinePbDataset import (
    load_episode,
    load_reference_pbs,
)
from robo_manip_baselines.misc.futureimagination.PlotOnlinePbDatasetAllObject import (
    REFERENCE_COLORS,
)
from robo_manip_baselines.misc.futureimagination.Wp4PlotStyle import PLOT_STYLE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Animate online PB from exactly one .rmb episode directory.",
    )
    parser.add_argument("rmb_file", type=Path, help="single .rmb episode directory")
    parser.add_argument("--output", type=Path, help="output MP4 path")
    return parser.parse_args()


def read_video_timing(path: Path) -> tuple[int, Fraction, Fraction]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_frames,avg_frame_rate,duration_ts,time_base",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stream = json.loads(result.stdout)["streams"][0]
    frames = int(stream["nb_frames"])
    fps = Fraction(stream["avg_frame_rate"])
    duration = int(stream["duration_ts"]) * Fraction(stream["time_base"])
    if frames <= 0 or fps <= 0 or duration != frames / fps:
        raise ValueError(f"Expected a constant-rate RMB camera video: {path}")
    return frames, fps, duration


@plt.rc_context(PLOT_STYLE)
def save_animation(
    episode: dict,
    reference_pbs: np.ndarray,
    fps: Fraction,
    output_path: Path,
) -> None:
    time = episode["plot_time"]
    pb = episode["plot_pb"]
    figure, axis = plt.subplots(figsize=(7, 4.8), layout="constrained")
    # Initialize with the whole trajectory to fix the limits before animation.
    (line,) = axis.plot(time, pb, color="black", linewidth=2.0)
    for object_id, reference_pb in enumerate(reference_pbs):
        axis.axhline(
            reference_pb,
            color=REFERENCE_COLORS[object_id],
            linestyle="--",
            linewidth=2.0,
        )
    axis.set_xlabel("Elapsed time [s]")
    axis.set_ylabel(r"$d$", fontsize=20)
    axis.set_title("Online parameter estimation")
    axis.grid(True)
    axis.margins(x=0.01, y=0.08)
    figure.canvas.draw()
    axis.set_autoscale_on(False)
    figure.set_layout_engine("none")
    line.set_data([], [])

    writer = FFMpegWriter(
        fps=fps,
        codec="libx264",
        metadata={"comment": f"WP4 checkpoint: {episode['source_checkpoint']}"},
        extra_args=[
            "-pix_fmt",
            "yuv420p",
            "-video_track_timescale",
            str(fps.numerator),
            "-movflags",
            "+faststart",
        ],
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with writer.saving(figure, str(output_path), dpi=150):
            for frame_idx in range(len(time)):
                line.set_data(time[: frame_idx + 1], pb[: frame_idx + 1])
                writer.grab_frame()
    finally:
        plt.close(figure)


def main() -> None:
    args = parse_args()
    rmb_file = args.rmb_file.resolve()
    if rmb_file.suffix != ".rmb" or not rmb_file.is_dir():
        raise ValueError(f"Specify exactly one .rmb episode directory: {rmb_file}")
    videos = sorted(rmb_file.glob("*.rmb.mp4"))
    if not videos:
        raise ValueError(f"No RMB camera MP4 found in {rmb_file}")
    timing = read_video_timing(videos[0])
    for video in videos[1:]:
        if read_video_timing(video) != timing:
            raise ValueError(f"Camera video timing differs: {videos[0]} and {video}")
    frames, fps, duration = timing

    episode = load_episode(str(rmb_file))
    if len(episode["plot_pb"]) != frames:
        raise ValueError("PB sample count must equal camera video frame count")
    if (
        not np.isfinite(episode["plot_pb"]).all()
        or not np.isfinite(episode["plot_time"]).all()
    ):
        raise ValueError("PB and timestamps must be finite")
    if np.any(np.diff(episode["plot_time"]) < 0):
        raise ValueError("Episode timestamps must be nondecreasing")
    checkpoint = episode["source_checkpoint"]
    if checkpoint.name != "policy_best.ckpt":
        raise ValueError(f"Expected the experiment's policy_best.ckpt: {checkpoint}")
    reference_pbs = load_reference_pbs(checkpoint, [0, 1, 2])
    output = (args.output or rmb_file / "online_pb_animation.mp4").resolve()
    if output.suffix != ".mp4" or output.name.endswith(".rmb.mp4"):
        raise ValueError(
            "Output must end in .mp4, but not the reserved .rmb.mp4 suffix"
        )
    if output.exists():
        raise FileExistsError(output)

    print(f"reference videos: {[str(video) for video in videos]}")
    print(f"reference checkpoint: {checkpoint}")
    print(
        f"frames: {frames}, fps: {fps}, duration: {float(duration):.6f} s", flush=True
    )
    save_animation(episode, reference_pbs, fps, output)
    if read_video_timing(output) != timing:
        raise RuntimeError(f"Output video timing does not match the source: {output}")
    print(output)


if __name__ == "__main__":
    main()
