"""Re-evaluate four DatasetUR5e Lifting conditions from saved RMB episodes.

Run from the repository root in the existing Python environment. Evaluation
directories are matched below; separate timestamped runs are never pooled.
The constant-PB paths match the ordinary rollout scripts. Their learning rate
may have been edited on the evaluation PC; directory names do not establish
the actual learning rate used for a saved run.
"""

import argparse
import csv
import re
from pathlib import Path

from robo_manip_baselines.misc.futureimagination.AnalyzeLiftingSuccess import (
    DEFAULT_LIFT_THRESHOLD_M,
    DEFAULT_TILT_THRESHOLD_DEG,
    analyze_episode,
    find_unique_rmb_files,
)

# Paths relative to --eval_root, matching the existing rollout scripts.
EVAL_PATTERNS = {
    "baseline_eef": "DatasetMujocoUR5eLiftingi/DP_eef_pose_baseline_eval_*",
    "baseline_eef_16obs": "DatasetMujocoUR5eLiftingi/DP_eef_pose_baseline_16obs_eval_*",
    "state_constant_pb": ("DatasetMujocoUR5eLiftingi/DP_eef_pose_constant_pb_eval_*"),
    "image_constant_pb": (
        "DatasetMujocoUR5eLiftingi_LeftScratchVAE9_Wp4JointPos_DpEefPose/"
        "DP_constant_pb_adam_eval_*"
    ),
}
TRAINING_SEEDS = (42, 52, 62)
OBJECT_IDS = (0, 1, 2, 4, 5, 6, 7)
SEEN_OBJECT_IDS = (0, 1, 2)
TRAINING_SEED_PATTERN = re.compile(r"trainseed(\d+)(?=[_/\.]|$)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval_root",
        type=Path,
        default=Path("robo_manip_baselines/dataset/tests/FutureImagination"),
        help="root containing the DatasetMujocoUR5eLiftingi evaluation directories",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="defaults to <eval_root>/lifting_success_by_seed",
    )
    return parser.parse_args()


def analyze_run(eval_dir: Path) -> list[dict]:
    rows = []
    seen = set()
    for filename in find_unique_rmb_files([eval_dir / "rmb"]):
        relative_path = Path(filename).relative_to(eval_dir)
        seeds = {
            int(seed) for seed in TRAINING_SEED_PATTERN.findall(str(relative_path))
        }
        if len(seeds) != 1 or not seeds.issubset(TRAINING_SEEDS):
            raise ValueError(f"Expected one training seed (42/52/62): {filename}")
        training_seed = seeds.pop()
        episode = analyze_episode(
            filename, DEFAULT_LIFT_THRESHOLD_M, DEFAULT_TILT_THRESHOLD_DEG
        )
        key = (training_seed, episode["group"], episode["world_idx"])
        if key in seen:
            raise ValueError(f"Duplicate seed/object/world {key}: {filename}")
        seen.add(key)
        rows.append({"training_seed": training_seed, **episode})
    return rows


def summarize_run(rows: list[dict]) -> list[dict]:
    expected = {
        (seed, f"I{object_id}", world_idx)
        for seed in TRAINING_SEEDS
        for object_id in OBJECT_IDS
        for world_idx in range(100 * object_id + 70, 100 * object_id + 80)
    }
    actual = {(r["training_seed"], r["group"], r["world_idx"]) for r in rows}
    if actual - expected:
        raise ValueError(
            f"Unexpected seed/object/world entries: {sorted(actual - expected)}"
        )

    groups = {
        "all": tuple(f"I{i}" for i in OBJECT_IDS),
        "seen": tuple(f"I{i}" for i in SEEN_OBJECT_IDS),
        "unseen": tuple(f"I{i}" for i in OBJECT_IDS if i not in SEEN_OBJECT_IDS),
        **{f"I{i}": (f"I{i}",) for i in OBJECT_IDS},
    }
    summaries = []
    for seed in (*TRAINING_SEEDS, "all"):
        for group, object_names in groups.items():
            selected = [
                r
                for r in rows
                if (seed == "all" or r["training_seed"] == seed)
                and r["group"] in object_names
            ]
            expected_count = (
                (len(TRAINING_SEEDS) if seed == "all" else 1) * len(object_names) * 10
            )
            total = len(selected)
            success = sum(r["success"] for r in selected)
            success_once = sum(r["success_once"] for r in selected)
            summaries.append(
                {
                    "training_seed": seed,
                    "group": group,
                    "total": total,
                    "expected_total": expected_count,
                    "complete": total == expected_count,
                    "success": success,
                    "success_rate": success / total if total else None,
                    "success_once": success_once,
                    "success_once_rate": success_once / total if total else None,
                    "lift_threshold_m": DEFAULT_LIFT_THRESHOLD_M,
                    "tilt_threshold_deg": DEFAULT_TILT_THRESHOLD_DEG,
                }
            )
    return summaries


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    runs = []
    for condition, pattern in EVAL_PATTERNS.items():
        directories = sorted(p for p in args.eval_root.glob(pattern) if p.is_dir())
        if not directories:
            raise FileNotFoundError(
                f"No evaluations for {condition}: {args.eval_root / pattern}"
            )
        runs.extend((condition, directory) for directory in directories)

    episodes = []
    summaries = []
    for condition, eval_dir in runs:
        print(f"\n{condition}: {eval_dir}", flush=True)
        metadata = {"condition": condition, "eval_dir": str(eval_dir.resolve())}
        rows = analyze_run(eval_dir)
        episodes.extend({**metadata, **row} for row in rows)
        for summary in summarize_run(rows):
            summaries.append({**metadata, **summary})
            if summary["group"] in ("all", "seen", "unseen"):
                total = summary["total"]
                rate = summary["success_rate"]
                rate_text = f"{100 * rate:.2f}%" if rate is not None else "N/A"
                status = "" if summary["complete"] else " INCOMPLETE"
                print(
                    f"  seed {summary['training_seed']} {summary['group']}: "
                    f"final {summary['success']}/{total} ({rate_text}), "
                    f"once {summary['success_once']}/{total}; "
                    f"expected {summary['expected_total']}{status}",
                    flush=True,
                )

    if not episodes:
        raise ValueError("No RMB episodes found in the evaluation directories")
    output_dir = args.output_dir or args.eval_root / "lifting_success_by_seed"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "per_episode.csv", episodes)
    write_csv(output_dir / "success_by_seed.csv", summaries)
    print(f"\nSaved: {output_dir.resolve()}")
    if any(not row["complete"] for row in summaries):
        print(
            "WARNING: incomplete runs are included; check complete and expected_total."
        )


if __name__ == "__main__":
    main()
