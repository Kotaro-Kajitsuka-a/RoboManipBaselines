import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from robo_manip_baselines.common import DataKey, RmbData, find_rmb_files

DEFAULT_NUM_OBJECTS = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Plot two-dimensional online PB trajectories.",
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
    parser.add_argument(
        "--material_object_ids",
        type=int,
        nargs="+",
        default=None,
        help="trained PB object IDs shown as references",
    )
    return parser.parse_args()


def load_episode(filename: str) -> dict:
    with RmbData(filename) as rmb_data:
        assert DataKey.TIME in rmb_data, filename
        assert DataKey.MATERIAL_PROPERTY in rmb_data, filename

        num_steps = rmb_data[DataKey.TIME].shape[0]
        pb = rmb_data[DataKey.MATERIAL_PROPERTY][:]
        if "online_pb_wp4_checkpoint" in rmb_data.attrs:
            source_checkpoint = Path(rmb_data.attrs["online_pb_wp4_checkpoint"])
        else:
            source_checkpoint = Path(
                rmb_data[DataKey.MATERIAL_PROPERTY].attrs["source_checkpoint"]
            )

    assert pb.shape == (num_steps, 2), (filename, pb.shape)
    return {
        "name": Path(filename).stem,
        "plot_pb": pb,
        "source_checkpoint": source_checkpoint,
    }


def get_source_checkpoint(episodes: list[dict]) -> Path:
    source_checkpoint = episodes[0]["source_checkpoint"]
    for episode in episodes[1:]:
        assert episode["source_checkpoint"] == source_checkpoint, episode["name"]
    return source_checkpoint


def load_reference_pb(
    checkpoint_path: Path,
    material_object_ids: list[int] | None,
) -> tuple[np.ndarray, list[int]]:
    assert checkpoint_path.is_file(), checkpoint_path
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    reference_pb = state_dict["material_property.weight"]
    assert reference_pb.ndim == 2, reference_pb.shape
    assert reference_pb.shape[1] == 2, reference_pb.shape
    if material_object_ids is None:
        material_object_ids = list(range(DEFAULT_NUM_OBJECTS))
    assert len(material_object_ids) == len(set(material_object_ids))
    assert all(
        0 <= object_id < reference_pb.shape[0] for object_id in material_object_ids
    )
    return reference_pb[material_object_ids].detach().numpy(), material_object_ids


def get_mean_trajectory(episodes: list[dict]) -> np.ndarray:
    common_length = min(len(episode["plot_pb"]) for episode in episodes)
    pb_array = np.stack([episode["plot_pb"][:common_length] for episode in episodes])
    return pb_array.mean(axis=0)


def get_default_output_path(dataset_path: Path) -> Path:
    return dataset_path.resolve() / "online_pb_trajectories_2d.png"


def save_plot(
    episodes: list[dict],
    reference_pb: np.ndarray,
    material_object_ids: list[int],
    output_path: Path,
) -> None:
    colors = plt.cm.tab10(np.arange(len(material_object_ids)))
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

    for episode_idx, episode in enumerate(episodes):
        trajectory = episode["plot_pb"]
        axis.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            color="tab:blue",
            linewidth=1.0,
            alpha=0.25,
            label=f"individual episodes (n={len(episodes)})"
            if episode_idx == 0
            else None,
        )
        axis.scatter(
            trajectory[-1, 0],
            trajectory[-1, 1],
            s=12,
            color="tab:blue",
            alpha=0.35,
        )

    mean_pb = get_mean_trajectory(episodes)
    axis.plot(
        mean_pb[:, 0],
        mean_pb[:, 1],
        color="black",
        linewidth=2.6,
        label=f"mean trajectory (all {len(episodes)} episodes)",
    )

    for color_idx, (object_id, value) in enumerate(
        zip(material_object_ids, reference_pb, strict=True)
    ):
        x_offset = -8 if value[0] >= label_x_center else 8
        y_offset = 8 if object_id in (1, 2, 4) else -8
        axis.scatter(
            value[0],
            value[1],
            s=140,
            color=colors[color_idx],
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
            color=colors[color_idx],
            fontweight="bold",
        )

    axis.set_xlim(x_center - half_range, x_center + half_range)
    axis.set_ylim(y_center - half_range, y_center + half_range)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("PB dimension 1")
    axis.set_ylabel("PB dimension 2")
    axis.set_title("Online PB trajectories")
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
    source_checkpoint = get_source_checkpoint(episodes)
    reference_pb, material_object_ids = load_reference_pb(
        source_checkpoint,
        args.material_object_ids,
    )

    output_path = args.output
    if output_path is None:
        output_path = get_default_output_path(args.dataset_path)
    save_plot(
        episodes,
        reference_pb,
        material_object_ids,
        output_path,
    )

    final_pb = np.stack([episode["plot_pb"][-1] for episode in episodes])
    print(f"episodes: {len(episodes)}")
    print(f"trained PBs: {reference_pb.tolist()}")
    print(
        f"final PB: mean={final_pb.mean(axis=0).tolist()}, "
        f"std={final_pb.std(axis=0).tolist()}"
    )
    print(output_path.resolve())


if __name__ == "__main__":
    main()
