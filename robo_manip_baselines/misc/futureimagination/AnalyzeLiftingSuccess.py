import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

from robo_manip_baselines.common import DataKey, RmbData, find_rmb_files


DEFAULT_LIFT_THRESHOLD_M = 0.10
DEFAULT_TILT_THRESHOLD_DEG = 7.5
WORLD_PATTERN = re.compile(r"_world(\d+)(?:_|$)")
OBJECT_PATTERNS = {
    "I0": ("WrenchPredObject0", "MujocoUR5eLiftingi_I0"),
    "I1": ("WrenchPredObject1", "MujocoUR5eLiftingi_I1"),
    "I2": ("WrenchPredObject2", "MujocoUR5eLiftingi_I2"),
    "I4": ("WrenchPredObject4", "MujocoUR5eLiftingi_I4"),
    "I5": ("WrenchPredObject5", "MujocoUR5eLiftingi_I5"),
    "I6": ("WrenchPredObject6", "MujocoUR5eLiftingi_I6"),
    "I7": ("WrenchPredObject7", "MujocoUR5eLiftingi_I7"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate final-state and success-once criteria for Lifting RMB data."
        )
    )
    parser.add_argument(
        "dataset_paths",
        type=Path,
        nargs="+",
        help="one or more RMB files or directories containing RMB data",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help=(
            "directory in which to save the per-episode CSV and summary JSON; "
            "defaults to the input dataset directory for a single input"
        ),
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="lifting_success",
        help="prefix of the output CSV and JSON files",
    )
    parser.add_argument(
        "--lift_threshold_m",
        type=float,
        default=DEFAULT_LIFT_THRESHOLD_M,
        help="minimum block lift for success [m]",
    )
    parser.add_argument(
        "--tilt_threshold_deg",
        type=float,
        default=DEFAULT_TILT_THRESHOLD_DEG,
        help="maximum block tilt for success [deg] (strict inequality)",
    )
    parser.add_argument(
        "--expected_episode_count",
        type=int,
        default=None,
        help="assert the number of discovered RMB episodes when specified",
    )
    return parser.parse_args()


def find_unique_rmb_files(dataset_paths: list[Path]) -> list[str]:
    filename_by_resolved_path = {}
    for dataset_path in dataset_paths:
        for filename in find_rmb_files(str(dataset_path)):
            resolved_path = str(Path(filename).resolve())
            assert resolved_path not in filename_by_resolved_path, resolved_path
            filename_by_resolved_path[resolved_path] = filename
    return list(filename_by_resolved_path.values())


def get_output_dir(dataset_paths: list[Path], output_dir: Path | None) -> Path:
    if output_dir is not None:
        return output_dir
    assert len(dataset_paths) == 1, (
        "--output_dir is required when multiple dataset paths are specified"
    )
    dataset_path = dataset_paths[0]
    if dataset_path.name.endswith(".rmb"):
        return dataset_path.parent
    if dataset_path.is_dir():
        return dataset_path
    return dataset_path.parent


def get_object_name(path: Path) -> str:
    path_text = str(path)
    matched_object_names = [
        object_name
        for object_name, patterns in OBJECT_PATTERNS.items()
        if object_name in path.parts
        or any(pattern in path_text for pattern in patterns)
    ]
    assert len(matched_object_names) == 1, (path, matched_object_names)
    return matched_object_names[0]


def get_group_name(path: Path) -> str:
    object_name = get_object_name(path)
    if "A_known_to_operator" in path.parts:
        return f"{object_name}_A"
    if "B_unknown_to_operator" in path.parts:
        return f"{object_name}_B"
    return object_name


def calculate_tilt_deg(quaternion_wxyz: np.ndarray) -> np.ndarray:
    quaternion_wxyz = np.asarray(quaternion_wxyz, dtype=np.float64)
    quaternion_norm = np.linalg.norm(quaternion_wxyz, axis=-1, keepdims=True)
    assert np.all(quaternion_norm > 0.0)
    quaternion_wxyz = quaternion_wxyz / quaternion_norm
    qw, qx, qy, qz = np.moveaxis(quaternion_wxyz, -1, 0)

    # The block's longitudinal axis is its local Y axis. Its world-Z
    # component is zero when the block is horizontal; yaw has no effect.
    local_y_world_z = 2.0 * (qy * qz + qw * qx)
    return np.rad2deg(np.arcsin(np.clip(np.abs(local_y_world_z), 0.0, 1.0)))


def analyze_episode(
    filename: str,
    lift_threshold_m: float,
    tilt_threshold_deg: float,
) -> dict:
    episode_path = Path(filename)
    world_match = WORLD_PATTERN.search(episode_path.name)
    assert world_match is not None, episode_path

    with RmbData(filename) as rmb_data:
        pose = np.asarray(rmb_data[DataKey.MEASURED_TBLOCK_POSE][:])
        time = np.asarray(rmb_data[DataKey.TIME][:], dtype=np.float64).reshape(-1)

    assert pose.ndim == 2 and pose.shape[1] == 7, (episode_path, pose.shape)
    assert len(pose) == len(time) and len(time) > 0, (
        episode_path,
        pose.shape,
        time.shape,
    )
    assert np.all(np.diff(time) >= 0.0), episode_path

    lift_m = pose[:, 2] - pose[0, 2]
    tilt_deg = calculate_tilt_deg(pose[:, 3:7])
    success_series = (lift_m >= lift_threshold_m) & (tilt_deg < tilt_threshold_deg)
    success_idxes = np.flatnonzero(success_series)
    success_once = len(success_idxes) > 0
    if success_once:
        first_success_time_s = float(time[success_idxes[0]] - time[0])
    else:
        first_success_time_s = None

    final_lift_m = float(lift_m[-1])
    final_tilt_deg = float(tilt_deg[-1])
    final_lift_success = final_lift_m >= lift_threshold_m
    final_tilt_success = final_tilt_deg < tilt_threshold_deg
    final_success = final_lift_success and final_tilt_success

    return {
        "group": get_group_name(episode_path),
        "world_idx": int(world_match.group(1)),
        "duration_s": float(time[-1] - time[0]),
        "final_lift_m": final_lift_m,
        "final_tilt_deg": final_tilt_deg,
        "final_lift_success": final_lift_success,
        "final_tilt_success": final_tilt_success,
        "success": final_success,
        "success_once": success_once,
        "first_success_time_s": first_success_time_s,
        "episode_path": str(episode_path.resolve()),
    }


def summarize_rows(rows: list[dict]) -> dict:
    success_count = sum(row["success"] for row in rows)
    success_once_count = sum(row["success_once"] for row in rows)
    first_success_times = [
        row["first_success_time_s"]
        for row in rows
        if row["first_success_time_s"] is not None
    ]
    return {
        "total": len(rows),
        "success": success_count,
        "failure": len(rows) - success_count,
        "success_rate": success_count / len(rows),
        "success_once": success_once_count,
        "never_succeeded": len(rows) - success_once_count,
        "success_once_rate": success_once_count / len(rows),
        "min_final_lift_m": min(row["final_lift_m"] for row in rows),
        "max_final_tilt_deg": max(row["final_tilt_deg"] for row in rows),
        "first_success_time_s": {
            "count": len(first_success_times),
            "min": min(first_success_times) if first_success_times else None,
            "mean": float(np.mean(first_success_times))
            if first_success_times
            else None,
            "max": max(first_success_times) if first_success_times else None,
        },
    }


def main() -> None:
    args = parse_args()
    assert args.lift_threshold_m >= 0.0, args.lift_threshold_m
    assert 0.0 <= args.tilt_threshold_deg <= 90.0, args.tilt_threshold_deg
    output_dir = get_output_dir(args.dataset_paths, args.output_dir)

    filenames = find_unique_rmb_files(args.dataset_paths)
    if len(filenames) == 0:
        raise ValueError(f"No RMB episodes found: {args.dataset_paths}")
    if args.expected_episode_count is not None:
        assert len(filenames) == args.expected_episode_count, (
            len(filenames),
            args.expected_episode_count,
        )

    rows = [
        analyze_episode(
            filename,
            args.lift_threshold_m,
            args.tilt_threshold_deg,
        )
        for filename in filenames
    ]
    group_order = {group: idx for idx, group in enumerate(OBJECT_PATTERNS)}
    rows.sort(
        key=lambda row: (
            group_order.get(row["group"].split("_")[0], len(group_order)),
            row["group"],
            row["world_idx"],
        )
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{args.output_prefix}_per_episode.csv"
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    group_names = sorted(
        {row["group"] for row in rows},
        key=lambda group: (
            group_order.get(group.split("_")[0], len(group_order)),
            group,
        ),
    )
    summary = {
        "success_criteria": {
            "final_lift_m_at_least": args.lift_threshold_m,
            "final_tilt_deg_less_than": args.tilt_threshold_deg,
        },
        "dataset_paths": [str(path.resolve()) for path in args.dataset_paths],
        **summarize_rows(rows),
        "groups": {
            group: summarize_rows([row for row in rows if row["group"] == group])
            for group in group_names
        },
    }
    summary_path = output_dir / f"{args.output_prefix}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(json.dumps(summary, indent=2))
    final_failures = [row for row in rows if not row["success"]]
    if final_failures:
        print("Final-state failures:")
        for row in final_failures:
            print(
                f"  {row['group']} world{row['world_idx']}: "
                f"lift={100.0 * row['final_lift_m']:.2f} cm, "
                f"tilt={row['final_tilt_deg']:.2f} deg, "
                f"success_once={row['success_once']}"
            )
    print(csv_path.resolve())
    print(summary_path.resolve())


if __name__ == "__main__":
    main()
