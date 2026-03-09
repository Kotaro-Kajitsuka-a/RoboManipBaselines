"""
Evaluate box-rotation episodes with TPR (throughput-with-regret).

TPR definition:
    TPR = (1 / N) * sum_i( (S_i / t_i) - ((1 - S_i) / T_MAX) )

where:
    - S_i = 1 if final angle is in [SUCCESS_MIN_DEGREE, SUCCESS_MAX_DEGREE], else 0
    - t_i = min(duration_i, T_MAX)
    - duration_i = time[-1] - time[0]
"""

import argparse
from pathlib import Path

import h5py
import numpy as np

SUCCESS_MIN_DEGREE = 85.0
SUCCESS_MAX_DEGREE = 95.0
T_MAX_SECONDS = 30.0

COLOR_CYAN_BOLD = "\033[1;36m"
COLOR_YELLOW_BOLD = "\033[1;33m"
COLOR_GREEN_BOLD = "\033[1;32m"
COLOR_RED_BOLD = "\033[1;31m"
COLOR_RESET = "\033[0m"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate box rotation with TPR from main_eval.hdf5 files."
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
    print(f"{COLOR_CYAN_BOLD}[EvalBoxRotationTPR] Found {count} main_eval.hdf5 files{COLOR_RESET}")
    if count == 0:
        raise FileNotFoundError(f"No main_eval.hdf5 found under: {dataset_dir}")

    tpr_terms = []
    success_count = 0
    success_durations = []

    for idx, hdf5_path in enumerate(hdf5_paths, start=1):
        with h5py.File(hdf5_path, "r") as h5file:
            if "time" not in h5file:
                raise KeyError(f"time key is missing in {hdf5_path}")
            if "box_z_degree" not in h5file:
                raise KeyError(f"box_z_degree key is missing in {hdf5_path}")

            time_seq = np.asarray(h5file["time"][:], dtype=np.float64)
            angle_seq = np.asarray(h5file["box_z_degree"][:], dtype=np.float64)

        if time_seq.shape[0] != angle_seq.shape[0]:
            raise AssertionError(
                f"time and box_z_degree length mismatch in {hdf5_path}: "
                f"{time_seq.shape[0]} vs {angle_seq.shape[0]}"
            )
        if time_seq.shape[0] == 0:
            raise AssertionError(f"empty sequence in {hdf5_path}")

        duration = float(time_seq[-1] - time_seq[0])
        if duration <= 0.0:
            raise AssertionError(f"non-positive duration in {hdf5_path}: {duration:.6f}")

        initial_angle = float(angle_seq[0])
        if duration >= T_MAX_SECONDS:
            final_angle = float(np.interp(T_MAX_SECONDS, time_seq, angle_seq))
        else:
            final_angle = float(angle_seq[-1])
        success = SUCCESS_MIN_DEGREE <= final_angle <= SUCCESS_MAX_DEGREE
        S_i = 1.0 if success else 0.0
        t_i = min(duration, T_MAX_SECONDS)
        tpr_i = (S_i / t_i) - ((1.0 - S_i) / T_MAX_SECONDS)
        tpr_terms.append(tpr_i)

        if success:
            success_count += 1
            success_durations.append(duration)
            color = COLOR_GREEN_BOLD
            status = "SUCCESS"
        else:
            color = COLOR_RED_BOLD
            status = "FAIL"

        print(
            f"{color}[{idx}/{count}] {status} initial={initial_angle:.3f} deg, final={final_angle:.3f} deg, "
            f"duration={duration:.3f} s, t_i={t_i:.3f} s, tpr_i={tpr_i:.6f} | {hdf5_path}{COLOR_RESET}"
        )

    tpr = float(np.mean(tpr_terms))
    atr = float(np.mean(success_durations)) if success_durations else float("nan")
    print(
        f"{COLOR_CYAN_BOLD}[EvalBoxRotationTPR] TPR={tpr:.6f}, "
        f"success={success_count}/{count}, ATR={atr:.6f}, t_max={T_MAX_SECONDS:.1f}s, "
        f"success_range=[{SUCCESS_MIN_DEGREE:.1f}, {SUCCESS_MAX_DEGREE:.1f}] deg{COLOR_RESET}"
    )


if __name__ == "__main__":
    main()
