"""
Evaluate box-rotation episodes with TPR (throughput-with-regret).

The first run creates ``distribution.csv`` under the dataset directory.
Fill its ``distribution`` column with ``ID`` or ``OOD``, then run this script
again to calculate metrics for ID, OOD, and all episodes.

TPR definition:
    TPR = (1 / N) * sum_i( (S_i / t_i) - ((1 - S_i) / T_MAX) )

where:
    - S_i = 1 if final angle is in [SUCCESS_MIN_DEGREE, SUCCESS_MAX_DEGREE], else 0
    - t_i = min(duration_i, T_MAX)
    - duration_i = time[-1] - time[0]
"""

import argparse
import csv
from pathlib import Path

import h5py
import numpy as np

SUCCESS_MIN_DEGREE = 85.0
SUCCESS_MAX_DEGREE = 95.0
T_MAX_SECONDS = 30.0
DISTRIBUTION_CSV_NAME = "distribution.csv"
DISTRIBUTION_CSV_COLUMNS = ["hdf5_path", "distribution"]
DISTRIBUTIONS = ("ID", "OOD")

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


def to_relative_path(hdf5_path: Path, dataset_dir: Path) -> str:
    return hdf5_path.relative_to(dataset_dir).as_posix()


def create_distribution_csv(
    csv_path: Path, dataset_dir: Path, hdf5_paths: list[Path]
) -> None:
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=DISTRIBUTION_CSV_COLUMNS)
        writer.writeheader()
        for hdf5_path in hdf5_paths:
            writer.writerow(
                {
                    "hdf5_path": to_relative_path(hdf5_path, dataset_dir),
                    "distribution": "",
                }
            )

    print(
        f"{COLOR_YELLOW_BOLD}[EvalBoxRotationTPR] Created {csv_path}\n"
        "Fill the distribution column with ID or OOD, then run this script again."
        f"{COLOR_RESET}"
    )


def load_distributions(
    csv_path: Path, dataset_dir: Path, hdf5_paths: list[Path]
) -> dict[str, str]:
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == DISTRIBUTION_CSV_COLUMNS, (
            f"Unexpected csv columns: {reader.fieldnames}. "
            f"Expected: {DISTRIBUTION_CSV_COLUMNS}"
        )
        rows = list(reader)

    distributions = {}
    for row in rows:
        relative_path = row["hdf5_path"]
        distribution = row["distribution"].strip()
        assert relative_path not in distributions, (
            f"Duplicate hdf5_path in {csv_path}: {relative_path}"
        )
        assert distribution in DISTRIBUTIONS, (
            f"distribution must be ID or OOD: "
            f"{relative_path} -> {distribution!r}"
        )
        distributions[relative_path] = distribution

    expected_paths = {
        to_relative_path(hdf5_path, dataset_dir) for hdf5_path in hdf5_paths
    }
    csv_paths = set(distributions)
    assert csv_paths == expected_paths, (
        f"{csv_path} does not match the discovered main_eval.hdf5 files. "
        f"Missing: {sorted(expected_paths - csv_paths)}, "
        f"Extra: {sorted(csv_paths - expected_paths)}"
    )
    return distributions


def calculate_metrics(results: list[dict]) -> dict:
    count = len(results)
    successful_results = [result for result in results if result["success"]]
    success_count = len(successful_results)
    success_rate = success_count / count if count else float("nan")
    atr = (
        float(np.mean([result["duration"] for result in successful_results]))
        if successful_results
        else float("nan")
    )
    tpr = (
        float(np.mean([result["tpr"] for result in results]))
        if results
        else float("nan")
    )
    return {
        "count": count,
        "success_count": success_count,
        "success_rate": success_rate,
        "atr": atr,
        "tpr": tpr,
    }


def print_summary(distribution: str, metrics: dict) -> None:
    print(
        f"{COLOR_CYAN_BOLD}[EvalBoxRotationTPR][{distribution}] "
        f"TPR={metrics['tpr']:.6f}, "
        f"SR={metrics['success_count']}/{metrics['count']} "
        f"({metrics['success_rate']:.2%}), "
        f"ATR={metrics['atr']:.6f}, t_max={T_MAX_SECONDS:.1f}s, "
        f"success_range=[{SUCCESS_MIN_DEGREE:.1f}, "
        f"{SUCCESS_MAX_DEGREE:.1f}] deg{COLOR_RESET}"
    )


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"dataset directory not found: {dataset_dir}")

    hdf5_paths = sorted(dataset_dir.glob("**/main_eval.hdf5"))
    count = len(hdf5_paths)
    print(
        f"{COLOR_CYAN_BOLD}[EvalBoxRotationTPR] Found {count} main_eval.hdf5 files{COLOR_RESET}"
    )
    if count == 0:
        raise FileNotFoundError(f"No main_eval.hdf5 found under: {dataset_dir}")

    distribution_csv_path = dataset_dir / DISTRIBUTION_CSV_NAME
    if not distribution_csv_path.exists():
        create_distribution_csv(distribution_csv_path, dataset_dir, hdf5_paths)
        return

    distributions = load_distributions(
        distribution_csv_path, dataset_dir, hdf5_paths
    )
    results = []

    for idx, hdf5_path in enumerate(hdf5_paths, start=1):
        relative_path = to_relative_path(hdf5_path, dataset_dir)
        distribution = distributions[relative_path]

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
            raise AssertionError(
                f"non-positive duration in {hdf5_path}: {duration:.6f}"
            )

        initial_angle = float(angle_seq[0])
        if duration >= T_MAX_SECONDS:
            final_angle = float(np.interp(T_MAX_SECONDS, time_seq, angle_seq))
        else:
            final_angle = float(angle_seq[-1])
        success = SUCCESS_MIN_DEGREE <= final_angle <= SUCCESS_MAX_DEGREE
        S_i = 1.0 if success else 0.0
        t_i = min(duration, T_MAX_SECONDS)
        tpr_i = (S_i / t_i) - ((1.0 - S_i) / T_MAX_SECONDS)
        results.append(
            {
                "distribution": distribution,
                "success": success,
                "duration": duration,
                "tpr": tpr_i,
            }
        )

        if success:
            color = COLOR_GREEN_BOLD
            status = "SUCCESS"
        else:
            color = COLOR_RED_BOLD
            status = "FAIL"

        print(
            f"{color}[{idx}/{count}][{distribution}] {status} "
            f"initial={initial_angle:.3f} deg, final={final_angle:.3f} deg, "
            f"duration={duration:.3f} s, t_i={t_i:.3f} s, "
            f"tpr_i={tpr_i:.6f} | {hdf5_path}{COLOR_RESET}"
        )

    for distribution in (*DISTRIBUTIONS, "ALL"):
        distribution_results = (
            results
            if distribution == "ALL"
            else [
                result
                for result in results
                if result["distribution"] == distribution
            ]
        )
        print_summary(distribution, calculate_metrics(distribution_results))


if __name__ == "__main__":
    main()
