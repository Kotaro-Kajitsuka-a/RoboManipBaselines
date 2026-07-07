"""
Evaluate box-rotation episodes from `main_eval.hdf5`.

Metric overview:
    score = (final-state success ratio) / (time to reach final state)

Definitions:
    - final state angle: `theta` at the last frame.
    - final-state success ratio:
        corrected_rotated_amount / required_amount
        where
            required_amount = 90 - initial_angle
            corrected_rotated_amount = max(0, required_amount - abs(final_angle - 90))
    - time to reach final state:
        the timestamp when `theta` last enters
        [final_angle - LAST_STATE_TOLERANCE, final_angle + LAST_STATE_TOLERANCE].
"""

import argparse
from pathlib import Path

import h5py
import numpy as np

LAST_STATE_TOLERANCE = 1.0
TARGET_ANGLE_DEGREE = 90.0

COLOR_CYAN_BOLD = "\033[1;36m"
COLOR_YELLOW_BOLD = "\033[1;33m"
COLOR_GREEN_BOLD = "\033[1;32m"
COLOR_RED_BOLD = "\033[1;31m"
COLOR_RESET = "\033[0m"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate box rotation episodes from main_eval.hdf5 files."
    )
    parser.add_argument("dataset_dir", type=str, help="Path to dataset directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"dataset directory not found: {dataset_dir}")

    hdf5_paths = sorted(dataset_dir.glob("**/main_eval.hdf5"))
    count = len(hdf5_paths)

    # Bold cyan for visibility in terminal.
    print(
        f"{COLOR_CYAN_BOLD}[EvalBoxRotation] Found {count} main_eval.hdf5 files{COLOR_RESET}"
    )

    if count == 0:
        raise FileNotFoundError(f"No main_eval.hdf5 found under: {dataset_dir}")
    if count != 10:
        print(
            f"{COLOR_RED_BOLD}[Warning] Expected 10 files, but found {count}. Continue anyway.{COLOR_RESET}"
        )

    score_list = []
    for idx, hdf5_path in enumerate(hdf5_paths, start=1):
        with h5py.File(hdf5_path, "r") as h5file:
            if "time" not in h5file:
                raise KeyError(f"time key is missing in {hdf5_path}")
            if "box_z_degree" not in h5file:
                raise KeyError(f"box_z_degree key is missing in {hdf5_path}")

            time_seq = np.asarray(h5file["time"][:], dtype=np.float64)
            box_z_degree_seq = np.asarray(h5file["box_z_degree"][:], dtype=np.float64)

        if time_seq.shape[0] != box_z_degree_seq.shape[0]:
            raise AssertionError(
                f"time and box_z_degree length mismatch in {hdf5_path}: "
                f"{time_seq.shape[0]} vs {box_z_degree_seq.shape[0]}"
            )
        if time_seq.shape[0] == 0:
            raise AssertionError(f"empty sequence in {hdf5_path}")

        initial_angle = float(box_z_degree_seq[0])
        final_angle = float(box_z_degree_seq[-1])

        print(
            f"{COLOR_YELLOW_BOLD}[{idx}/{count}] initial={initial_angle:.3f} deg, "
            f"final={final_angle:.3f} deg | {hdf5_path}{COLOR_RESET}"
        )

        required_amount = TARGET_ANGLE_DEGREE - initial_angle
        if required_amount <= 1e-8:
            raise AssertionError(
                f"required_amount must be positive in {hdf5_path} (initial_angle={initial_angle:.6f})"
            )
        corrected_rotated_amount = max(
            0.0, required_amount - abs(final_angle - TARGET_ANGLE_DEGREE)
        )
        success_ratio = corrected_rotated_amount / required_amount

        in_range = np.abs(box_z_degree_seq - final_angle) <= LAST_STATE_TOLERANCE
        # Find start index of the trailing in-range segment (last entry to final-state band).
        entry_idx = int(len(in_range) - 1)
        while entry_idx >= 0 and bool(in_range[entry_idx]):
            entry_idx -= 1
        entry_idx += 1

        reach_time = float(time_seq[entry_idx] - time_seq[0])
        if reach_time <= 0.0:
            raise AssertionError(
                f"reach_time must be positive in {hdf5_path} (reach_time={reach_time:.6f}, entry_idx={entry_idx})"
            )

        score = success_ratio / reach_time
        score_list.append(score)
        print(
            f"{COLOR_GREEN_BOLD}  final={final_angle:.3f} deg, reach_time={reach_time:.3f} s, "
            f"success_ratio={success_ratio:.6f}, score={score:.6f}{COLOR_RESET}"
        )

    avg_score = float(np.mean(score_list))
    print(
        f"{COLOR_CYAN_BOLD}[EvalBoxRotation] Average score over {count} episodes: {avg_score:.6f}{COLOR_RESET}"
    )


if __name__ == "__main__":
    main()
