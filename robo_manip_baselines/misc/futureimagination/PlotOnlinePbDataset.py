import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from robo_manip_baselines.common import DataKey, RmbData, find_rmb_files


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


def load_episode(filename: str) -> dict:
    with RmbData(filename) as rmb_data:
        assert DataKey.TIME in rmb_data, filename
        assert DataKey.MATERIAL_PROPERTY in rmb_data, filename

        time = rmb_data[DataKey.TIME][:]
        pb = rmb_data[DataKey.MATERIAL_PROPERTY][:]
        if "online_pb_wp4_checkpoint" in rmb_data.attrs:
            source_checkpoint = Path(rmb_data.attrs["online_pb_wp4_checkpoint"])
        else:
            source_checkpoint = Path(
                rmb_data[DataKey.MATERIAL_PROPERTY].attrs["source_checkpoint"]
            )

    assert time.ndim == 1, (filename, time.shape)
    assert pb.shape == (len(time), 1), (filename, pb.shape)
    elapsed_time = time - time[0]
    return {
        "name": Path(filename).stem,
        "plot_time": elapsed_time,
        "plot_pb": pb[:, 0],
        "source_checkpoint": source_checkpoint,
    }


def get_mean_trajectory(episodes: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    common_length = min(len(episode["plot_pb"]) for episode in episodes)
    time_array = np.stack(
        [episode["plot_time"][:common_length] for episode in episodes]
    )
    pb_array = np.stack([episode["plot_pb"][:common_length] for episode in episodes])
    return time_array.mean(axis=0), pb_array.mean(axis=0)


def load_reference_pbs(checkpoint_path: Path) -> np.ndarray:
    assert checkpoint_path.is_file(), checkpoint_path
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    reference_pbs = state_dict["material_property.weight"]
    assert reference_pbs.ndim == 2 and reference_pbs.shape[1] == 1, reference_pbs.shape
    assert reference_pbs.shape[0] >= 3, reference_pbs.shape
    return reference_pbs[:3, 0].numpy()


def get_source_checkpoint(episodes: list[dict]) -> Path:
    source_checkpoint = episodes[0]["source_checkpoint"]
    for episode in episodes[1:]:
        assert episode["source_checkpoint"] == source_checkpoint, episode["name"]
    return source_checkpoint


def get_default_output_path(dataset_path: Path) -> Path:
    return dataset_path.resolve() / "online_pb_trajectories.png"


def save_plot(
    episodes: list[dict],
    reference_pbs: np.ndarray,
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
    reference_colors = ("tab:red", "tab:orange", "tab:green")
    for object_id, (reference_pb, color) in enumerate(
        zip(reference_pbs, reference_colors, strict=True)
    ):
        axis.axhline(
            reference_pb,
            color=color,
            linestyle="--",
            linewidth=2.0,
            label=f"trained Object{object_id} PB={reference_pb:.4f}",
        )

    axis.set_xlabel("episode elapsed time [s]")
    axis.set_ylabel("PB")
    axis.set_title("Online PB trajectories")
    axis.grid(alpha=0.3)
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
    reference_pbs = load_reference_pbs(source_checkpoint)

    output_path = args.output
    if output_path is None:
        output_path = get_default_output_path(args.dataset_path)
    save_plot(
        episodes,
        reference_pbs,
        output_path,
    )

    final_pb = np.asarray([episode["plot_pb"][-1] for episode in episodes])
    print(f"episodes: {len(episodes)}")
    print(f"trained PBs: {reference_pbs.tolist()}")
    print(
        f"final PB: mean={final_pb.mean():.6f}, std={final_pb.std():.6f}, "
        f"range=[{final_pb.min():.6f}, {final_pb.max():.6f}]"
    )
    print(output_path.resolve())


if __name__ == "__main__":
    main()
