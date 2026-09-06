#!/usr/bin/env python
import re
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

DATASET_DIR = Path("robo_manip_baselines/dataset/DatasetPushingLine")
OUTPUT_PATH = DATASET_DIR / "initial_fz_vs_episode_index.png"


def episode_index(rmb_path: Path) -> int:
    match = re.search(r"_world\d+_(\d+)\.rmb$", rmb_path.name)
    return int(match.group(1))


def initial_fz(rmb_path: Path) -> float:
    with h5py.File(rmb_path / "main.rmb.hdf5") as h5file:
        return float(np.mean(h5file["measured_eef_wrench"][:5, 2]))


def load_points(rmb_paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    points = sorted([(episode_index(path), initial_fz(path)) for path in rmb_paths])
    return np.asarray([point[0] for point in points]), np.asarray(
        [point[1] for point in points]
    )


def correlation(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(x, y)[0, 1])


def main() -> None:
    training_paths = []
    for object_id in (2, 3, 4):
        training_paths.extend(
            (DATASET_DIR / "training" / f"WrenchPredObject{object_id}" / "fast").glob(
                "*.rmb"
            )
        )

    validation_paths = []
    for object_dir in sorted((DATASET_DIR / "validation").glob("WrenchPredObject*")):
        for rmb_path in object_dir.glob("*.rmb"):
            if object_dir.name == "WrenchPredObject4" and episode_index(rmb_path) == 0:
                continue
            validation_paths.append(rmb_path)

    train_x, train_y = load_points(training_paths)
    val_x, val_y = load_points(validation_paths)
    episode_x = np.concatenate([train_x, val_x])
    episode_y = np.concatenate([train_y, val_y])
    slope, intercept = np.polyfit(episode_x, episode_y, 1)
    fit_x = np.asarray([episode_x.min(), episode_x.max()])

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.scatter(
        episode_x,
        episode_y,
        s=28,
        label=f"episodes (r={correlation(episode_x, episode_y):.3f})",
    )
    ax.plot(fit_x, slope * fit_x + intercept, "--", label="linear fit")

    ax.axhline(0.0, color="gray", linewidth=1.0)
    ax.set_xlabel("Episode index (world0_XXX)")
    ax.set_ylabel("Mean Fz over steps 0-4")
    ax.set_title("Initial wrench shift over consecutive episodes")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=180)
    plt.close(fig)
    print(OUTPUT_PATH.resolve())


if __name__ == "__main__":
    main()
