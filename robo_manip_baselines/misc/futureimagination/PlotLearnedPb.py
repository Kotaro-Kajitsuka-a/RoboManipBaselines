import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

DEFAULT_MATERIAL_OBJECT_IDS = (0, 1, 2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot learned one-dimensional PBs in a WP4 checkpoint."
    )
    parser.add_argument(
        "checkpoint",
        type=Path,
        help="WP4 checkpoint containing material_property.weight",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="output PNG path (default: next to the checkpoint)",
    )
    parser.add_argument(
        "--material_object_ids",
        type=int,
        nargs="+",
        default=list(DEFAULT_MATERIAL_OBJECT_IDS),
        help="trained PB object IDs shown in the plot",
    )
    return parser.parse_args()


def load_pb(
    checkpoint_path: Path,
    material_object_ids: list[int] | tuple[int, ...] | None = None,
) -> np.ndarray:
    checkpoint_path = checkpoint_path.resolve()
    assert checkpoint_path.is_file(), checkpoint_path

    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    material_property = state_dict["material_property.weight"]
    assert material_property.ndim == 2, material_property.shape
    assert material_property.shape[1] == 1, material_property.shape

    if material_object_ids is None:
        material_object_ids = DEFAULT_MATERIAL_OBJECT_IDS
    assert len(material_object_ids) == len(set(material_object_ids))
    assert all(
        0 <= object_id < material_property.shape[0] for object_id in material_object_ids
    )
    return material_property[list(material_object_ids), 0].detach().numpy()


def get_default_output_path(checkpoint_path: Path) -> Path:
    return checkpoint_path.resolve().with_name(
        f"{checkpoint_path.stem}_learned_pb_1d.png"
    )


def save_plot(
    pb: np.ndarray,
    output_path: Path,
    material_object_ids: list[int] | tuple[int, ...] | None = None,
) -> None:
    if material_object_ids is None:
        material_object_ids = DEFAULT_MATERIAL_OBJECT_IDS
    assert len(pb) == len(material_object_ids), (pb.shape, material_object_ids)

    colors = plt.cm.tab10(np.arange(len(material_object_ids)))
    value_range = float(np.ptp(pb))
    padding = max(value_range * 0.2, 0.05)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(10, 3.2))
    axis.axhline(0.0, color="0.45", linewidth=1.5)

    for color_idx, (object_id, value) in enumerate(
        zip(material_object_ids, pb, strict=True)
    ):
        axis.scatter(value, 0.0, s=110, color=colors[color_idx], zorder=2)
        axis.annotate(
            f"WrenchPredObject{object_id}\nPB={value:.4f}",
            (value, 0.0),
            xytext=(0, 18),
            textcoords="offset points",
            ha="center",
            va="bottom",
            color=colors[color_idx],
        )

    axis.set_xlim(float(pb.min()) - padding, float(pb.max()) + padding)
    axis.set_ylim(-0.18, 0.45)
    axis.set_xlabel("PB")
    axis.set_yticks([])
    axis.set_title("Learned one-dimensional material PBs")
    axis.grid(axis="x", alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    pb = load_pb(args.checkpoint, args.material_object_ids)
    output_path = args.output
    if output_path is None:
        output_path = get_default_output_path(args.checkpoint)

    save_plot(pb, output_path, args.material_object_ids)

    for object_id, value in zip(args.material_object_ids, pb, strict=True):
        print(f"WrenchPredObject{object_id}: {value:.6f}")
    print(output_path.resolve())


if __name__ == "__main__":
    main()
