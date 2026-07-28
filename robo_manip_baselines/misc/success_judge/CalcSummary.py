#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

T_MAX_SECONDS = 5.0
DISTRIBUTION_COLUMN = "distribution"
DISTRIBUTIONS = ("ID", "OOD")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate Success Rate, ATR, and TPR from task_completion.csv"
    )
    parser.add_argument("csv_path", type=Path)
    return parser.parse_args()


def calculate_metrics(rows):
    total_count = len(rows)
    success_count = 0
    success_times = []
    tpr_terms = []

    for row in rows:
        success = row["success"] == "1"

        if success:
            duration = float(row["completion_time"])
            t_i = min(duration, T_MAX_SECONDS)
            tpr_i = 1.0 / t_i

            success_count += 1
            success_times.append(duration)
        else:
            tpr_i = -1.0 / T_MAX_SECONDS

        tpr_terms.append(tpr_i)

    success_rate = success_count / total_count if total_count else float("nan")
    atr = sum(success_times) / len(success_times) if success_times else float("nan")
    tpr = sum(tpr_terms) / len(tpr_terms) if tpr_terms else float("nan")

    return {
        "total_count": total_count,
        "success_count": success_count,
        "success_rate": success_rate,
        "atr": atr,
        "tpr": tpr,
    }


def make_summary_row(name, metrics):
    return {
        "rmb_path": "__summary__" if name == "ALL" else f"__summary_{name}__",
        "camera": "",
        "completion_frame": "",
        "completion_time": (
            f"ATR={metrics['atr']:.6f},TPR={metrics['tpr']:.6f}"
        ),
        "success": (
            f"{metrics['success_count']}/{metrics['total_count']} "
            f"({metrics['success_rate']:.2%})"
        ),
        DISTRIBUTION_COLUMN: name,
    }


def print_metrics(name, metrics):
    print(f"[{name}]")
    print(f"Trials        : {metrics['total_count']}")
    print(f"Success       : {metrics['success_count']}")
    print(f"Success Rate  : {metrics['success_rate']:.2%}")
    print(f"ATR           : {metrics['atr']:.6f} s")
    print(f"TPR           : {metrics['tpr']:.6f}")


def main():
    args = parse_args()

    with args.csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames)

    assert fieldnames[-1] == DISTRIBUTION_COLUMN, (
        f"The last csv column must be '{DISTRIBUTION_COLUMN}': {fieldnames}"
    )

    # 以前追加した summary を削除
    rows = [row for row in rows if not row["rmb_path"].startswith("__summary")]

    for row in rows:
        row[DISTRIBUTION_COLUMN] = row[DISTRIBUTION_COLUMN].strip()
        assert row[DISTRIBUTION_COLUMN] in DISTRIBUTIONS, (
            f"distribution must be ID or OOD: "
            f"{row['rmb_path']} -> {row[DISTRIBUTION_COLUMN]!r}"
        )

    metrics_by_distribution = {
        distribution: calculate_metrics(
            [
                row
                for row in rows
                if row[DISTRIBUTION_COLUMN] == distribution
            ]
        )
        for distribution in DISTRIBUTIONS
    }
    metrics_by_distribution["ALL"] = calculate_metrics(rows)

    for distribution in (*DISTRIBUTIONS, "ALL"):
        rows.append(
            make_summary_row(
                distribution,
                metrics_by_distribution[distribution],
            )
        )

    with args.csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    for distribution in (*DISTRIBUTIONS, "ALL"):
        print_metrics(distribution, metrics_by_distribution[distribution])
        print()
    print(f"T_MAX         : {T_MAX_SECONDS:.1f} s")


if __name__ == "__main__":
    main()
