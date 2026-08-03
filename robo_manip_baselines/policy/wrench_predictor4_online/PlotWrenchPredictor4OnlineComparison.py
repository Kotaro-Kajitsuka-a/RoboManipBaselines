import argparse
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "result_dirs",
        type=Path,
        nargs="+",
        help="WrenchPredictor4Online result directories to compare",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="output PNG path (default: next to the result directories)",
    )
    return parser.parse_args()


def get_label(result_dir):
    match = re.search(r"world(\d+)_object(\d+)", result_dir.name, re.IGNORECASE)
    if match is None:
        return result_dir.name
    return f"world{match.group(1)} / Object{match.group(2)}"


def load_result(result_dir):
    csv_path = result_dir / "pb_adaptation.csv"
    pb_path = result_dir / "adapted_pb.pt"
    assert csv_path.is_file(), csv_path
    assert pb_path.is_file(), pb_path

    with csv_path.open() as file:
        rows = list(csv.DictReader(file))
    result = torch.load(pb_path, map_location="cpu", weights_only=True)
    reference_pb = result["reference_pb"].numpy()
    assert reference_pb.shape[1] == 1, reference_pb.shape
    return {
        "label": get_label(result_dir),
        "steps": np.asarray([int(row["adaptation_step"]) for row in rows]),
        "pb": np.asarray([float(row["pb"]) for row in rows]),
        "reference_pb": reference_pb[:, 0],
        "lr": result.get("lr"),
    }


def main():
    args = parse_args()
    results = [load_result(result_dir) for result_dir in args.result_dirs]
    reference_pb = results[0]["reference_pb"]
    for result in results[1:]:
        assert np.allclose(reference_pb, result["reference_pb"])

    output_path = args.output
    if output_path is None:
        output_path = args.result_dirs[0].parent / "pb_adaptation_comparison.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(10, 6))
    colors = plt.get_cmap("tab10").colors
    for result_idx, result in enumerate(results):
        axis.plot(
            result["steps"],
            result["pb"],
            color=colors[result_idx % len(colors)],
            linewidth=2.2,
            label=f"{result['label']}: final={result['pb'][-1]:.4f}",
        )

    for object_id, pb in enumerate(reference_pb):
        axis.axhline(
            pb,
            color=colors[object_id % len(colors)],
            linestyle="--",
            linewidth=1.2,
            alpha=0.8,
            label=f"trained PB{object_id}={pb:.4f}",
        )

    axis.set_xlabel("adaptation step")
    axis.set_ylabel("PB")
    learning_rates = {result["lr"] for result in results}
    title = "Online PB adaptation from the center of trained PBs"
    if len(learning_rates) == 1 and None not in learning_rates:
        title += f" (lr={learning_rates.pop():g})"
    axis.set_title(title)
    axis.grid(alpha=0.3)
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    print(output_path.resolve())


if __name__ == "__main__":
    main()
