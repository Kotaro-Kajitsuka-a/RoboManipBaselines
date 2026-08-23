#!/usr/bin/env python3

import argparse
from pathlib import Path

import h5py
import numpy as np
import yaml

DEFAULT_SUCCESS_REWARD = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read saved rollout RMB data and report the first successful step "
            "and elapsed time for every episode."
        )
    )
    parser.add_argument(
        "rmb_path",
        type=Path,
        help="RMB file or directory containing saved rollout RMB episodes",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "output YAML path; defaults to success_timing.yaml next to an "
            "input rmb directory, or inside another input directory"
        ),
    )
    parser.add_argument(
        "--success_reward",
        type=float,
        default=DEFAULT_SUCCESS_REWARD,
        help="minimum reward treated as success",
    )
    parser.add_argument(
        "--expected_episode_count",
        type=int,
        default=None,
        help="assert the number of discovered RMB episodes when specified",
    )
    return parser.parse_args()


def find_rmb_paths(path: Path) -> list[Path]:
    if path.name.endswith(".rmb") or path.suffix.lower() == ".hdf5":
        paths = [path]
    elif path.is_dir():
        paths = [candidate for candidate in path.rglob("*.rmb") if candidate.is_dir()]
        paths.extend(
            candidate
            for candidate in path.rglob("*.hdf5")
            if candidate.is_file() and not candidate.name.endswith(".rmb.hdf5")
        )
    else:
        raise ValueError(f"RMB path not found: {path}")
    return sorted(paths)


def get_hdf5_path(rmb_path: Path) -> Path:
    if rmb_path.suffix.lower() == ".hdf5":
        return rmb_path
    return rmb_path / "main.rmb.hdf5"


def get_string_attr(h5file: h5py.File, key: str) -> str | None:
    if key not in h5file.attrs:
        return None
    value = h5file.attrs[key]
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.generic):
        value = value.item()
    return str(value)


def round_float(value: float) -> float:
    return round(float(value), 9)


def analyze_episode(rmb_path: Path, success_reward: float) -> dict:
    hdf5_path = get_hdf5_path(rmb_path)
    assert hdf5_path.is_file(), hdf5_path

    with h5py.File(hdf5_path, "r") as h5file:
        assert "time" in h5file, rmb_path
        assert "reward" in h5file, rmb_path
        time = np.asarray(h5file["time"][:], dtype=np.float64).reshape(-1)
        reward = np.asarray(h5file["reward"][:], dtype=np.float64).reshape(-1)
        demo_name = get_string_attr(h5file, "demo_name")
        env_name = get_string_attr(h5file, "env")
        world_idx = h5file.attrs.get("world_idx")

    assert len(time) > 0, rmb_path
    assert len(time) == len(reward), (rmb_path, time.shape, reward.shape)
    assert np.all(np.diff(time) >= 0.0), rmb_path

    if isinstance(world_idx, np.generic):
        world_idx = world_idx.item()
    if world_idx is not None:
        world_idx = int(world_idx)

    success_idxes = np.flatnonzero(reward >= success_reward)
    success_once = len(success_idxes) > 0
    if success_once:
        first_success_step_index = int(success_idxes[0])
        first_success_time_s = round_float(time[first_success_step_index] - time[0])
    else:
        first_success_step_index = None
        first_success_time_s = None

    return {
        "demo_name": demo_name,
        "env": env_name,
        "world_idx": world_idx,
        "episode_length_steps": len(time),
        "duration_s": round_float(time[-1] - time[0]),
        "success_once": success_once,
        "success_last": bool(reward[-1] >= success_reward),
        "first_success_step_index": first_success_step_index,
        "first_success_time_s": first_success_time_s,
        "final_reward": round_float(reward[-1]),
        "rmb_path": str(rmb_path.resolve()),
    }


def summarize(episodes: list[dict]) -> dict:
    success_once_count = sum(episode["success_once"] for episode in episodes)
    success_last_count = sum(episode["success_last"] for episode in episodes)
    first_success_steps = [
        episode["first_success_step_index"]
        for episode in episodes
        if episode["first_success_step_index"] is not None
    ]
    first_success_times = [
        episode["first_success_time_s"]
        for episode in episodes
        if episode["first_success_time_s"] is not None
    ]

    return {
        "episode_count": len(episodes),
        "success_once_count": success_once_count,
        "success_once_rate": round_float(success_once_count / len(episodes)),
        "success_last_count": success_last_count,
        "success_last_rate": round_float(success_last_count / len(episodes)),
        "first_success_step_index": {
            "count": len(first_success_steps),
            "min": min(first_success_steps) if first_success_steps else None,
            "mean": round_float(np.mean(first_success_steps))
            if first_success_steps
            else None,
            "max": max(first_success_steps) if first_success_steps else None,
        },
        "first_success_time_s": {
            "count": len(first_success_times),
            "min": min(first_success_times) if first_success_times else None,
            "mean": round_float(np.mean(first_success_times))
            if first_success_times
            else None,
            "max": max(first_success_times) if first_success_times else None,
        },
    }


def get_output_path(rmb_path: Path, output: Path | None) -> Path:
    if output is not None:
        return output
    if rmb_path.is_dir() and rmb_path.name == "rmb":
        return rmb_path.parent / "success_timing.yaml"
    if rmb_path.is_dir() and not rmb_path.name.endswith(".rmb"):
        return rmb_path / "success_timing.yaml"
    return rmb_path.parent / f"{rmb_path.stem}_success_timing.yaml"


def main() -> None:
    args = parse_args()
    assert args.success_reward >= 0.0, args.success_reward

    rmb_paths = find_rmb_paths(args.rmb_path)
    if len(rmb_paths) == 0:
        raise ValueError(f"No RMB episodes found: {args.rmb_path}")
    if args.expected_episode_count is not None:
        assert len(rmb_paths) == args.expected_episode_count, (
            len(rmb_paths),
            args.expected_episode_count,
        )

    episodes = [
        analyze_episode(rmb_path, args.success_reward) for rmb_path in rmb_paths
    ]
    episodes.sort(
        key=lambda episode: (
            episode["demo_name"] or "",
            episode["world_idx"] if episode["world_idx"] is not None else -1,
            episode["rmb_path"],
        )
    )

    demo_names = sorted(
        {episode["demo_name"] for episode in episodes},
        key=lambda demo_name: demo_name or "",
    )
    output_data = {
        "source_path": str(args.rmb_path.resolve()),
        "success_criterion": {
            "data_key": "reward",
            "minimum_reward": args.success_reward,
            "comparison": ">=",
        },
        "first_success_step_index_base": 0,
        "summary": summarize(episodes),
        "groups_by_demo_name": {
            demo_name if demo_name is not None else "unknown": summarize(
                [episode for episode in episodes if episode["demo_name"] == demo_name]
            )
            for demo_name in demo_names
        },
        "episodes": episodes,
    }

    output_path = get_output_path(args.rmb_path, args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as file:
        yaml.safe_dump(output_data, file, sort_keys=False)

    for episode in episodes:
        label = episode["demo_name"] or Path(episode["rmb_path"]).name
        if episode["world_idx"] is not None:
            label += f" world{episode['world_idx']}"
        if episode["success_once"]:
            print(
                f"{label}: success at step {episode['first_success_step_index']} "
                f"({episode['first_success_time_s']:.3f} s)"
            )
        else:
            print(f"{label}: failure")
    print(
        f"Success once: {output_data['summary']['success_once_count']} / "
        f"{output_data['summary']['episode_count']}"
    )
    print(output_path.resolve())


if __name__ == "__main__":
    main()
