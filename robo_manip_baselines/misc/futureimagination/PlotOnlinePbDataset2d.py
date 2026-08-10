import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from robo_manip_baselines.common import DataKey, RmbData, find_rmb_files


NUM_OBJECTS = 5
INITIAL_OBJECT_ID = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Plot all two-dimensional online PB trajectories from WrenchPredObject0."
        ),
    )
    parser.add_argument(
        "dataset_path",
        type=Path,
        help="dataset directory containing online-PB RMB episodes",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="output PNG path",
    )
    return parser.parse_args()


def get_actual_object_id(filename: str) -> int:
    match = re.search(r"WrenchPredObject(\d+)", filename)
    assert match is not None, filename
    object_id = int(match.group(1))
    assert 0 <= object_id < NUM_OBJECTS, (filename, object_id)
    return object_id


def load_episode(filename: str) -> dict:
    with RmbData(filename) as rmb_data:
        assert DataKey.TIME in rmb_data, filename
        assert DataKey.MATERIAL_PROPERTY in rmb_data, filename

        num_steps = rmb_data[DataKey.TIME].shape[0]
        pb = rmb_data[DataKey.MATERIAL_PROPERTY][:]
        attrs = dict(rmb_data[DataKey.MATERIAL_PROPERTY].attrs)

    assert pb.shape == (num_steps, 2), (filename, pb.shape)
    skip = int(attrs["online_skip"])
    horizon = int(attrs["online_horizon"])
    num_updates = int(attrs["online_num_updates"])
    assert num_updates > 0, (filename, num_updates)
    first_update_idx = (horizon - 1) * skip
    update_idxes = first_update_idx + np.arange(num_updates) * skip
    assert update_idxes[-1] < num_steps, (filename, update_idxes[-1], num_steps)
    plot_idxes = np.concatenate([np.asarray([0]), update_idxes])

    initial_pb = np.asarray(attrs["initial_pb"], dtype=np.float64)
    assert initial_pb.shape == (2,), (filename, initial_pb.shape)
    return {
        "name": Path(filename).stem,
        "actual_object_id": get_actual_object_id(filename),
        "plot_pb": pb[plot_idxes],
        "final_pb": pb[-1],
        "initial_pb": initial_pb,
        "initial_object_id": int(attrs["initial_object_id"]),
        "initial_object_key": str(attrs["initial_object_key"]),
        "learning_rate": float(attrs["online_learning_rate"]),
        "source_checkpoint": Path(attrs["source_checkpoint"]),
    }


def check_episode_metadata(
    episodes: list[dict],
) -> tuple[np.ndarray, float, Path, str]:
    initial_pb = episodes[0]["initial_pb"]
    learning_rate = episodes[0]["learning_rate"]
    source_checkpoint = episodes[0]["source_checkpoint"]
    initial_object_key = episodes[0]["initial_object_key"]

    for episode in episodes:
        assert episode["initial_object_id"] == INITIAL_OBJECT_ID, episode["name"]
        assert episode["initial_object_key"] == initial_object_key, episode["name"]
        assert np.allclose(episode["initial_pb"], initial_pb), episode["name"]
        assert np.isclose(episode["learning_rate"], learning_rate), episode["name"]
        assert episode["source_checkpoint"] == source_checkpoint, episode["name"]

    return initial_pb, learning_rate, source_checkpoint, initial_object_key


def load_reference_pb(checkpoint_path: Path) -> np.ndarray:
    assert checkpoint_path.is_file(), checkpoint_path
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    reference_pb = state_dict["material_property.weight"]
    assert reference_pb.ndim == 2, reference_pb.shape
    assert reference_pb.shape[0] >= NUM_OBJECTS, reference_pb.shape
    assert reference_pb.shape[1] == 2, reference_pb.shape
    return reference_pb[:NUM_OBJECTS].detach().numpy()


def get_mean_trajectory(episodes: list[dict]) -> np.ndarray:
    common_length = min(len(episode["plot_pb"]) for episode in episodes)
    pb_array = np.stack([episode["plot_pb"][:common_length] for episode in episodes])
    return pb_array.mean(axis=0)


def get_default_output_path(dataset_path: Path) -> Path:
    return dataset_path.resolve() / "online_pb_trajectories_2d.png"


def save_plot(
    episodes: list[dict],
    reference_pb: np.ndarray,
    initial_object_key: str,
    learning_rate: float,
    output_path: Path,
) -> None:
    episodes_by_object = {
        object_id: [
            episode for episode in episodes if episode["actual_object_id"] == object_id
        ]
        for object_id in range(NUM_OBJECTS)
    }
    colors = plt.cm.tab10(np.arange(NUM_OBJECTS))
    all_pb = np.concatenate(
        [reference_pb] + [episode["plot_pb"] for episode in episodes],
        axis=0,
    )
    x_center = float(all_pb[:, 0].min() + all_pb[:, 0].max()) / 2.0
    y_center = float(all_pb[:, 1].min() + all_pb[:, 1].max()) / 2.0
    label_x_center = float(reference_pb[:, 0].min() + reference_pb[:, 0].max()) / 2.0
    value_range = max(float(np.ptp(all_pb[:, 0])), float(np.ptp(all_pb[:, 1])))
    half_range = max(value_range * 0.65, 0.05)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(9, 8))
    axis.axhline(0.0, color="0.7", linewidth=1.0)
    axis.axvline(0.0, color="0.7", linewidth=1.0)

    for object_id, object_episodes in episodes_by_object.items():
        for episode in object_episodes:
            trajectory = episode["plot_pb"]
            axis.plot(
                trajectory[:, 0],
                trajectory[:, 1],
                color=colors[object_id],
                linewidth=0.8,
                alpha=0.18,
            )
            axis.scatter(
                trajectory[-1, 0],
                trajectory[-1, 1],
                s=12,
                color=colors[object_id],
                alpha=0.35,
            )

        if object_episodes:
            mean_pb = get_mean_trajectory(object_episodes)
            axis.plot(
                mean_pb[:, 0],
                mean_pb[:, 1],
                color=colors[object_id],
                linewidth=2.6,
                label=f"Object{object_id} mean (n={len(object_episodes)})",
            )

    for object_id, value in enumerate(reference_pb):
        x_offset = -8 if value[0] >= label_x_center else 8
        y_offset = 8 if object_id in (1, 2, 4) else -8
        axis.scatter(
            value[0],
            value[1],
            s=140,
            color=colors[object_id],
            edgecolors="black",
            linewidths=0.8,
            zorder=4,
        )
        axis.annotate(
            f"WrenchPredObject{object_id}",
            value,
            xytext=(x_offset, y_offset),
            textcoords="offset points",
            ha="left" if x_offset > 0 else "right",
            va="bottom" if y_offset > 0 else "top",
            color=colors[object_id],
            fontweight="bold",
        )

    axis.set_xlim(x_center - half_range, x_center + half_range)
    axis.set_ylim(y_center - half_range, y_center + half_range)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("PB dimension 1")
    axis.set_ylabel("PB dimension 2")
    axis.set_title(
        f"Online PB trajectories from {initial_object_key} "
        f"(episodes={len(episodes)}, lr={learning_rate:g})"
    )
    axis.grid(alpha=0.25)
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    filenames = sorted(find_rmb_files(str(args.dataset_path)))
    assert len(filenames) > 0, args.dataset_path

    episodes = [load_episode(filename) for filename in filenames]
    initial_pb, learning_rate, source_checkpoint, initial_object_key = (
        check_episode_metadata(episodes)
    )
    reference_pb = load_reference_pb(source_checkpoint)
    assert np.allclose(initial_pb, reference_pb[INITIAL_OBJECT_ID]), (
        initial_pb,
        reference_pb[INITIAL_OBJECT_ID],
    )

    output_path = args.output
    if output_path is None:
        output_path = get_default_output_path(args.dataset_path)
    save_plot(
        episodes,
        reference_pb,
        initial_object_key,
        learning_rate,
        output_path,
    )

    print(f"episodes: {len(episodes)}")
    print(f"initial object: {initial_object_key}, PB={initial_pb.tolist()}")
    for object_id in range(NUM_OBJECTS):
        object_episodes = [
            episode for episode in episodes if episode["actual_object_id"] == object_id
        ]
        if not object_episodes:
            continue
        final_pb = np.stack([episode["final_pb"] for episode in object_episodes])
        print(
            f"WrenchPredObject{object_id}: episodes={len(object_episodes)}, "
            f"final mean={final_pb.mean(axis=0).tolist()}, "
            f"std={final_pb.std(axis=0).tolist()}"
        )
    print(output_path.resolve())


if __name__ == "__main__":
    main()
