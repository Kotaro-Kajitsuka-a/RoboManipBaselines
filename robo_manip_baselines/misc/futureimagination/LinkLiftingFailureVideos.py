import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path

from robo_manip_baselines.common import DataKey, RmbData

SEED_PATTERN = re.compile(r"(?:^|[/_])(?:train)?seed(\d+)(?:[/_]|$)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create symlinks to the left-camera RGB videos of failed Lifting "
            "episodes listed by AnalyzeLiftingSuccess.py. Also create online-PB "
            "plots for episodes that contain a material-property representation."
        )
    )
    parser.add_argument(
        "per_episode_csv",
        type=Path,
        help="per-episode CSV produced by AnalyzeLiftingSuccess.py",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        nargs="?",
        default=None,
        help=(
            "directory in which to create failure video symlinks; by default, "
            "create <CSV prefix>_failure_left_videos next to the input CSV"
        ),
    )
    parser.add_argument(
        "--camera_name",
        default="left",
        help="RGB camera whose MP4 is linked",
    )
    return parser.parse_args()


def parse_bool(value: str) -> bool:
    assert value in ("True", "False"), value
    return value == "True"


def get_seed(episode_path: Path) -> int:
    match = SEED_PATTERN.search(episode_path.as_posix())
    assert match is not None, episode_path
    return int(match.group(1))


def has_material_property(episode_path: Path) -> bool:
    with RmbData(str(episode_path)) as rmb_data:
        return DataKey.MATERIAL_PROPERTY in rmb_data


def main() -> None:
    args = parse_args()
    assert args.per_episode_csv.is_file(), args.per_episode_csv
    if args.output_dir is None:
        csv_prefix = args.per_episode_csv.stem.removesuffix("_per_episode")
        args.output_dir = (
            args.per_episode_csv.parent / f"{csv_prefix}_failure_left_videos"
        )

    with args.per_episode_csv.open(newline="") as file:
        rows = list(csv.DictReader(file))

    required_columns = {"group", "world_idx", "success", "episode_path"}
    assert rows, args.per_episode_csv
    assert required_columns <= rows[0].keys(), rows[0].keys()

    failure_rows = [row for row in rows if not parse_bool(row["success"])]
    created_count = 0
    existing_count = 0
    plot_count = 0
    skipped_plot_count = 0
    for row in failure_rows:
        episode_path = Path(row["episode_path"]).resolve()
        video_path = episode_path / f"{args.camera_name}_rgb_image.rmb.mp4"
        assert video_path.is_file(), video_path

        seed = get_seed(episode_path)
        world_idx = int(row["world_idx"])
        link_path = (
            args.output_dir / f"seed{seed}" / row["group"] / f"world{world_idx:03d}.mp4"
        )
        link_path.parent.mkdir(parents=True, exist_ok=True)
        relative_target = Path(os.path.relpath(video_path, link_path.parent))

        if link_path.is_symlink():
            assert Path(os.readlink(link_path)) == relative_target, link_path
            existing_count += 1
        else:
            assert not link_path.exists(), link_path
            link_path.symlink_to(relative_target)
            created_count += 1

        if has_material_property(episode_path):
            plot_path = link_path.with_suffix(".png")
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("PlotOnlinePbDataset.py")),
                    str(episode_path),
                    "--output",
                    str(plot_path),
                ],
                check=True,
            )
            plot_count += 1
        else:
            skipped_plot_count += 1

    print(f"Failure episodes: {len(failure_rows)} / {len(rows)}")
    print(f"Created symlinks: {created_count}")
    print(f"Existing symlinks: {existing_count}")
    print(f"Generated online PB plots: {plot_count}")
    print(f"Skipped online PB plots without material property: {skipped_plot_count}")
    print(args.output_dir.resolve())


if __name__ == "__main__":
    main()
