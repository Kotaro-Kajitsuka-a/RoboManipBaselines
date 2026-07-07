#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

T_MAX_SECONDS = 5.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate Success Rate, ATR, and TPR from task_completion.csv"
    )
    parser.add_argument("csv_path", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()

    with args.csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames)

    # 以前追加した summary を削除
    rows = [r for r in rows if r["rmb_path"] != "__summary__"]

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

    summary = {
        "rmb_path": "__summary__",
        "camera": "",
        "completion_frame": "",
        "completion_time": (f"ATR={atr:.6f},TPR={tpr:.6f}"),
        "success": (f"{success_count}/{total_count} ({success_rate:.2%})"),
    }

    rows.append(summary)

    with args.csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Trials        : {total_count}")
    print(f"Success       : {success_count}")
    print(f"Success Rate  : {success_rate:.2%}")
    print(f"ATR           : {atr:.6f} s")
    print(f"TPR           : {tpr:.6f}")
    print(f"T_MAX         : {T_MAX_SECONDS:.1f} s")


if __name__ == "__main__":
    main()
