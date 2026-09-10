"""Overlay seen-object online parameter trajectories, colored by true object."""

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from robo_manip_baselines.common import find_rmb_files
from robo_manip_baselines.misc.futureimagination.PlotOnlinePbDataset import (
    get_source_checkpoint,
    load_episode,
    load_reference_pbs,
)
from robo_manip_baselines.misc.futureimagination.Wp4PlotStyle import PLOT_STYLE

# Match the reference-line colors in PlotOnlinePbDataset.
REFERENCE_COLORS = (
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
    "tab:olive",
    "tab:cyan",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Overlay online parameter trajectories from WrenchPredObject<N> "
            "anywhere in episode paths. Negative and fractional object IDs are excluded."
        ),
    )
    parser.add_argument(
        "dataset_path", type=Path, help="dataset directory searched recursively"
    )
    parser.add_argument("--output", type=Path, default=None, help="output PNG path")
    parser.add_argument(
        "--reference_object_ids",
        type=int,
        nargs="+",
        default=None,
        help=(
            "seen object IDs to plot and show as references; by default, use all "
            "nonnegative integer object IDs found in paths plus references 0, 1, 2"
        ),
    )
    return parser.parse_args()


def load_object_episodes(
    dataset_path: Path, reference_object_ids: list[int] | None
) -> tuple[dict[int, list[dict]], list[int]]:
    object_filenames = {}
    for filename in sorted(find_rmb_files(str(dataset_path.absolute()))):
        # Capture fractional IDs as a whole so Object0_5 is not treated as Object0.
        object_tokens = set(
            re.findall(r"WrenchPredObject(-?\d+(?:[_.]\d+)?)", filename)
        )
        if not object_tokens:
            continue
        if any(not token.isdecimal() for token in object_tokens):
            print(f"Skip unseen object episode: {filename}")
            continue
        object_ids = {int(token) for token in object_tokens}
        if len(object_ids) != 1:
            raise ValueError(f"Ambiguous object IDs {sorted(object_ids)} in {filename}")
        object_id = object_ids.pop()
        object_filenames.setdefault(object_id, []).append(filename)

    if reference_object_ids is None:
        reference_object_ids = sorted({0, 1, 2} | object_filenames.keys())
    assert {0, 1, 2}.issubset(reference_object_ids), reference_object_ids
    assert len(reference_object_ids) == len(set(reference_object_ids))
    assert all(object_id >= 0 for object_id in reference_object_ids)
    reference_object_ids = sorted(reference_object_ids)

    object_episodes = {}
    for object_id, filenames in sorted(object_filenames.items()):
        if object_id not in reference_object_ids:
            print(f"Skip unselected object ID: {object_id}")
            continue
        object_episodes[object_id] = [load_episode(f) for f in filenames]
    assert object_episodes, f"No seen-object RMB episodes found under {dataset_path}"
    return object_episodes, reference_object_ids


@plt.rc_context(PLOT_STYLE)
def save_plot(
    object_episodes: dict[int, list[dict]],
    reference_pbs: np.ndarray,
    reference_object_ids: list[int],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7, 4.8), layout="constrained")
    for object_id, episodes in sorted(object_episodes.items()):
        for episode in episodes:
            axis.plot(
                episode["plot_time"],
                episode["plot_pb"],
                color=REFERENCE_COLORS[object_id % len(REFERENCE_COLORS)],
                linewidth=1.1,
                alpha=0.3,
            )

    for object_id, reference_pb in zip(
        reference_object_ids, reference_pbs, strict=True
    ):
        axis.axhline(
            reference_pb,
            color=REFERENCE_COLORS[object_id % len(REFERENCE_COLORS)],
            linestyle="--",
            linewidth=2.0,
            label=rf"$d_{{{object_id}}} = {reference_pb:.4f}$",
        )
    axis.set_xlabel("Elapsed time [s]")
    axis.set_ylabel(r"$d$", fontsize=20)
    axis.set_title("Online parameter estimation")
    axis.grid(True)
    axis.margins(x=0.01, y=0.08)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=14)
    figure.savefig(output_path)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    object_episodes, reference_object_ids = load_object_episodes(
        args.dataset_path, args.reference_object_ids
    )
    episodes = [episode for group in object_episodes.values() for episode in group]
    checkpoint = get_source_checkpoint(episodes)
    assert checkpoint.name == "policy_best.ckpt", checkpoint
    reference_pbs = load_reference_pbs(checkpoint, reference_object_ids)
    output_path = args.output
    if output_path is None:
        output_path = (
            args.dataset_path.resolve() / "online_pb_trajectories_all_object.png"
        )
    save_plot(object_episodes, reference_pbs, reference_object_ids, output_path)

    print(f"reference checkpoint: {checkpoint}")
    for object_id, group in object_episodes.items():
        final_pb = np.asarray([episode["plot_pb"][-1] for episode in group])
        print(
            f"Object{object_id}: episodes={len(group)}, "
            f"final d mean={final_pb.mean():.6f}, std={final_pb.std():.6f}, "
            f"range=[{final_pb.min():.6f}, {final_pb.max():.6f}]"
        )
    print(output_path.resolve())


if __name__ == "__main__":
    main()
