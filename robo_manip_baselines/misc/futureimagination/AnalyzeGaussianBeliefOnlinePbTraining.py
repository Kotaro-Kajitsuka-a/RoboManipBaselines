import argparse
import csv
import html
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from robo_manip_baselines.policy.wrench_predictor4_online.AddOnlinePbToDataset import (
    adapt_pb_trajectory,
)
from robo_manip_baselines.policy.wrench_predictor4_online.WrenchPredictor4OnlineUtils import (
    load_model_meta_info,
    load_pb,
    load_pb_table,
    load_policy,
)

from .AnalyzeGaussianBeliefOnlinePb import load_episode_context


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot mean Gaussian-belief Online PB over 25 training episodes."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--initial_object_id", type=int, default=0)
    parser.add_argument("--initial_std", type=float, default=0.25)
    parser.add_argument("--num_points", type=int, default=9)
    parser.add_argument("--beta", type=float, default=10.0)
    return parser.parse_args()


def summarize(records):
    final_mean = np.asarray([record["final_mean"] for record in records])
    final_std = np.asarray([record["final_std"] for record in records])
    target = records[0]["target_pb"]
    return {
        "final_mean": final_mean.mean(),
        "cross_episode_std": final_mean.std(),
        "rmse": np.sqrt(np.mean(np.square(final_mean - target))),
        "belief_std": final_std.mean(),
        "coverage_2std": np.mean(np.abs(final_mean - target) <= 2.0 * final_std),
        "updates": np.mean([record["num_updates"] for record in records]),
        "motion_time": np.mean([record["motion_time_s"] for record in records]),
    }


def plot_mean_trajectories(path, records_by_object, beta, num_points, reference_pbs):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    reference_colors = ("#d62728", "#f28e2b", "#2ca02c")
    for object_id, ax in enumerate(axes):
        records = records_by_object[object_id]
        assert len(records) == 25, (object_id, len(records))
        for episode_idx, record in enumerate(records):
            ax.plot(
                record["time"],
                record["mean_trajectory"],
                color="#1f5a94",
                linewidth=1.15,
                alpha=0.38,
                label="episode belief mean μ" if episode_idx == 0 else None,
            )
        for reference_id, (reference_pb, color) in enumerate(
            zip(reference_pbs, reference_colors, strict=True)
        ):
            ax.axhline(
                reference_pb,
                color=color,
                linestyle="--",
                linewidth=1.4,
                label=f"trained I{reference_id} PB",
            )
        mean_motion_time = np.mean([record["motion_time_s"] for record in records])
        ax.axvline(
            mean_motion_time,
            color="#2ca02c",
            linestyle=":",
            linewidth=1.3,
            label="mean 5 mm motion time",
        )
        ax.set_title(f"Object{object_id}: 25 training episodes")
        ax.set_xlabel("time [s]")
        ax.grid(alpha=0.24)
    axes[0].set_ylabel("PB")
    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle(
        f"25 training Online PB mean trajectories: beta={beta:g}, "
        f"{num_points}-point Gauss-Hermite, every overlapping window"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def save_csv(path, records):
    fieldnames = [
        "object_id",
        "episode",
        "target_pb",
        "num_updates",
        "final_mean",
        "final_std",
        "final_error",
        "covered_by_2std",
        "motion_time_s",
    ]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record[key] for key in fieldnames})


def write_html(path, args, initial_pb, pb_table, summaries):
    rows = []
    for object_id in range(3):
        summary = summaries[object_id]
        rows.append(
            "<tr>"
            f"<td>Object{object_id}</td>"
            f"<td>{pb_table[object_id, 0]:.5f}</td>"
            f"<td>{summary['final_mean']:.5f} ± "
            f"{summary['cross_episode_std']:.5f}</td>"
            f"<td>{summary['rmse']:.5f}</td>"
            f"<td>{summary['belief_std']:.5f}</td>"
            f"<td>{summary['coverage_2std'] * 100:.0f}%</td>"
            f"<td>{summary['updates']:.1f}</td>"
            "</tr>"
        )
    checkpoint = html.escape(str(args.checkpoint.resolve()))
    dataset = html.escape(str(args.dataset_dir.resolve()))
    path.write_text(
        f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Gaussian-belief Online PB training mean trajectories</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:1180px;margin:32px auto;padding:0 20px;line-height:1.65;color:#20242a}}
h1,h2{{line-height:1.25}} code{{background:#f1f3f5;padding:.12em .35em;border-radius:4px}}
.warn{{background:#fff4df;border-left:4px solid #e89522;padding:12px 16px}}
table{{border-collapse:collapse;width:100%;font-size:.92rem}} th,td{{border:1px solid #ccd2d8;padding:7px 9px;text-align:right}}
th:first-child,td:first-child{{text-align:left}} img{{width:100%;height:auto;border:1px solid #d8dde3}}
.small{{color:#5a6470;font-size:.9rem}}
</style></head><body>
<h1>Gaussian-belief Online PB — training mean trajectories</h1>
<p class="warn"><strong>注意:</strong> これはWP4の学習に使われたtraining data上の再生です。held-out性能ではなく、学習済み予測器が既知trajectory上でどのようにPB beliefを更新するかを見るfit診断です。</p>
<h2>条件</h2>
<ul>
<li>checkpoint: <code>{checkpoint}</code>（policy_best）</li>
<li>training: <code>{dataset}</code></li>
<li>Object0/1/2 各25 episode、合計75 episode。各episodeのGaussian平均PB μ(t)を1本ずつ表示。</li>
<li>初期belief: N({initial_pb:.6f}, {args.initial_std:.3f}²)、β={args.beta:g}、M={args.num_points}</li>
<li>従来backprop Online PBと同じく、最初のwindow完成後は全overlapping windowで更新。</li>
</ul>
<h2>各Object 25本のbelief平均 μ(t)</h2>
<img src="training_mean_trajectories.png" alt="25 Online PB mean trajectories per training object">
<p class="small">青い各線が1 episodeのGaussian belief平均 μ(t)です。25 episodeを平均した1本の曲線ではありません。σ帯は表示していません。緑点線は25本の5 mm motion時刻の平均です。</p>
<h2>episode終端の集計</h2>
<table><thead><tr><th>object</th><th>target PB</th><th>final μ mean ± episode std</th><th>RMSE</th><th>mean final σ</th><th>target coverage</th><th>mean updates</th></tr></thead><tbody>
{''.join(rows)}
</tbody></table>
<p class="small">終端表は各episode自身の終端を使っています。Raw records: <a href="training_episode_summary.csv">training_episode_summary.csv</a></p>
</body></html>""",
        encoding="utf-8",
    )


def main():
    args = parse_args()
    assert torch.cuda.is_available(), "This training replay must run on GPU."
    filenames = sorted(args.dataset_dir.glob("WrenchPredObject[0-2]/*.rmb"))
    assert len(filenames) == 75, len(filenames)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")
    model_meta_info = load_model_meta_info(args.checkpoint)
    policy = load_policy(args.checkpoint, model_meta_info, device)
    initial_pb, _ = load_pb(
        args.checkpoint,
        args.initial_object_id,
        model_meta_info,
    )
    pb_table, _ = load_pb_table(args.checkpoint, model_meta_info)

    records = []
    records_by_object = defaultdict(list)
    for episode_idx, filename in enumerate(filenames, start=1):
        object_id = int(filename.parent.name.removeprefix("WrenchPredObject"))
        time, motion_idx = load_episode_context(filename)
        mean_trajectory, std_trajectory, num_updates, final_mean, final_std = (
            adapt_pb_trajectory(
                str(filename),
                initial_pb,
                policy,
                model_meta_info,
                device,
                learning_rate=0.006,
                wrench_loss_weight=0.0,
                update_type="gaussian_belief",
                initial_std=args.initial_std,
                num_points=args.num_points,
                beta=args.beta,
            )
        )
        mean_trajectory = mean_trajectory[:, 0]
        std_trajectory = std_trajectory[:, 0]
        motion_time = time[motion_idx] if motion_idx is not None else np.nan
        record = {
            "object_id": object_id,
            "episode": filename.stem,
            "target_pb": float(pb_table[object_id, 0]),
            "num_updates": num_updates,
            "final_mean": float(final_mean[0]),
            "final_std": float(final_std[0]),
            "final_error": float(final_mean[0] - pb_table[object_id, 0]),
            "covered_by_2std": bool(
                abs(final_mean[0] - pb_table[object_id, 0]) <= 2.0 * final_std[0]
            ),
            "motion_time_s": float(motion_time),
            "time": time,
            "mean_trajectory": mean_trajectory,
            "std_trajectory": std_trajectory,
        }
        records.append(record)
        records_by_object[object_id].append(record)
        print(
            f"[{episode_idx:02d}/75] {filename.stem}: "
            f"mean={final_mean[0]:.5f}, std={final_std[0]:.5f}, "
            f"updates={num_updates}"
        )

    summaries = {
        object_id: summarize(records_by_object[object_id]) for object_id in range(3)
    }
    save_csv(args.output_dir / "training_episode_summary.csv", records)
    plot_mean_trajectories(
        args.output_dir / "training_mean_trajectories.png",
        records_by_object,
        args.beta,
        args.num_points,
        pb_table[:3, 0],
    )
    write_html(
        args.output_dir / "gaussian_belief_training_mean.html",
        args,
        float(initial_pb[0]),
        pb_table,
        summaries,
    )
    print(f"report: {args.output_dir / 'gaussian_belief_training_mean.html'}")


if __name__ == "__main__":
    main()
