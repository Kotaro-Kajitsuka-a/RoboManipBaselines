import argparse
import re
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from robo_manip_baselines.common import find_rmb_files
from robo_manip_baselines.policy.wrench_predictor4_online.AddConstantPbToDataset import (
    DATA_KEY,
    get_hdf5_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Plot the online PB trajectories stored in every RMB episode under "
            "one B dataset directory."
        ),
    )
    parser.add_argument(
        "dataset_path",
        type=Path,
        help="B dataset directory containing RMB episodes",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="output PNG path",
    )
    return parser.parse_args()


def get_actual_object_id(dataset_path: Path) -> int:
    match = re.search(r"WrenchPredObject(\d+)", str(dataset_path.resolve()))
    assert match is not None, (
        f"dataset path must contain WrenchPredObject<id>: {dataset_path}"
    )
    return int(match.group(1))


def load_episode(filename: str) -> dict:
    hdf5_path = get_hdf5_path(filename)
    with h5py.File(hdf5_path, "r") as h5file:
        assert "time" in h5file, hdf5_path
        assert DATA_KEY in h5file, hdf5_path

        time = h5file["time"][:]
        pb = h5file[DATA_KEY][:]
        attrs = dict(h5file[DATA_KEY].attrs)

    assert time.ndim == 1, (hdf5_path, time.shape)
    assert pb.shape == (len(time), 1), (hdf5_path, pb.shape)
    skip = int(attrs["online_skip"])
    horizon = int(attrs["online_horizon"])
    num_updates = int(attrs["online_num_updates"])
    first_update_idx = (horizon - 1) * skip
    update_idxes = first_update_idx + np.arange(num_updates) * skip
    assert update_idxes[-1] < len(time), (
        hdf5_path,
        update_idxes[-1],
        len(time),
    )
    plot_idxes = np.concatenate([np.asarray([0, first_update_idx - 1]), update_idxes])
    elapsed_time = time - time[0]
    return {
        "name": Path(filename).stem,
        "time": elapsed_time,
        "pb": pb[:, 0],
        "plot_time": elapsed_time[plot_idxes],
        "plot_pb": pb[plot_idxes, 0],
        "initial_pb": float(np.asarray(attrs["initial_pb"]).item()),
        "learning_rate": float(attrs["online_learning_rate"]),
        "source_checkpoint": Path(attrs["source_checkpoint"]),
    }


def check_episode_metadata(episodes: list[dict]) -> tuple[float, float, Path]:
    initial_pb = episodes[0]["initial_pb"]
    learning_rate = episodes[0]["learning_rate"]
    source_checkpoint = episodes[0]["source_checkpoint"]
    for episode in episodes[1:]:
        assert np.isclose(episode["initial_pb"], initial_pb), episode["name"]
        assert np.isclose(episode["learning_rate"], learning_rate), episode["name"]
        assert episode["source_checkpoint"] == source_checkpoint, episode["name"]
    return initial_pb, learning_rate, source_checkpoint


def load_reference_pb(checkpoint_path: Path, object_id: int) -> float:
    assert checkpoint_path.is_file(), checkpoint_path
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    reference_pb = state_dict["material_property.weight"]
    assert reference_pb.ndim == 2 and reference_pb.shape[1] == 1, reference_pb.shape
    assert object_id < reference_pb.shape[0], (object_id, reference_pb.shape)
    return reference_pb[object_id, 0].item()


def get_mean_trajectory(episodes: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    common_length = min(len(episode["plot_pb"]) for episode in episodes)
    time_array = np.stack(
        [episode["plot_time"][:common_length] for episode in episodes]
    )
    pb_array = np.stack([episode["plot_pb"][:common_length] for episode in episodes])
    return time_array.mean(axis=0), pb_array.mean(axis=0)


def get_default_output_path(dataset_path: Path) -> Path:
    return dataset_path.resolve() / "online_pb_trajectories.png"


def save_plot(
    episodes: list[dict],
    actual_object_id: int,
    initial_pb: float,
    target_pb: float,
    learning_rate: float,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(11, 6.5))

    for episode_idx, episode in enumerate(episodes):
        axis.plot(
            episode["plot_time"],
            episode["plot_pb"],
            color="tab:blue",
            linewidth=1.0,
            alpha=0.25,
            label=f"individual episodes (n={len(episodes)})"
            if episode_idx == 0
            else None,
        )

    mean_time, mean_pb = get_mean_trajectory(episodes)
    axis.plot(
        mean_time,
        mean_pb,
        color="black",
        linewidth=2.6,
        label=f"mean trajectory (all {len(episodes)} episodes)",
    )
    axis.axhline(
        target_pb,
        color="tab:red",
        linestyle="--",
        linewidth=2.0,
        label=f"trained Object{actual_object_id} PB={target_pb:.4f}",
    )
    if not np.isclose(initial_pb, target_pb):
        axis.axhline(
            initial_pb,
            color="0.4",
            linestyle=":",
            linewidth=2.0,
            label=f"initial PB={initial_pb:.4f}",
        )

    axis.set_xlabel("episode elapsed time [s]")
    axis.set_ylabel("PB")
    axis.set_title(
        f"Online PB identification: actual WrenchPredObject{actual_object_id} "
        f"(lr={learning_rate:g})"
    )
    axis.grid(alpha=0.3)
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    filenames = sorted(find_rmb_files(str(args.dataset_path)))
    assert len(filenames) > 0, args.dataset_path

    actual_object_id = get_actual_object_id(args.dataset_path)
    episodes = [load_episode(filename) for filename in filenames]
    initial_pb, learning_rate, source_checkpoint = check_episode_metadata(episodes)
    target_pb = load_reference_pb(source_checkpoint, actual_object_id)

    output_path = args.output
    if output_path is None:
        output_path = get_default_output_path(args.dataset_path)
    save_plot(
        episodes,
        actual_object_id,
        initial_pb,
        target_pb,
        learning_rate,
        output_path,
    )

    final_pb = np.asarray([episode["pb"][-1] for episode in episodes])
    print(f"episodes: {len(episodes)}")
    print(f"actual object: WrenchPredObject{actual_object_id}")
    print(f"initial PB: {initial_pb:.6f}")
    print(f"trained target PB: {target_pb:.6f}")
    print(
        f"final PB: mean={final_pb.mean():.6f}, std={final_pb.std():.6f}, "
        f"range=[{final_pb.min():.6f}, {final_pb.max():.6f}]"
    )
    print(output_path.resolve())


if __name__ == "__main__":
    main()
