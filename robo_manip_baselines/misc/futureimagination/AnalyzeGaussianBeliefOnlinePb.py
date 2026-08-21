import argparse
import csv
import html
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from robo_manip_baselines.common import DataKey, RmbData
from robo_manip_baselines.policy.wrench_predictor4_online.AddOnlinePbToDataset import (
    adapt_pb_trajectory,
)
from robo_manip_baselines.policy.wrench_predictor4_online.WrenchPredictor4OnlineUtils import (
    load_model_meta_info,
    load_pb,
    load_pb_table,
    load_policy,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Replay Gaussian-belief Online PB on WP4 validation episodes."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--initial_object_id", type=int, default=0)
    parser.add_argument("--initial_std", type=float, default=0.25)
    parser.add_argument("--num_points", type=int, default=9)
    parser.add_argument(
        "--beta_list",
        type=float,
        nargs="+",
        default=[0.3, 1.0, 3.0, 10.0, 30.0],
    )
    parser.add_argument("--selected_beta", type=float, default=10.0)
    return parser.parse_args()


def load_episode_context(filename):
    with RmbData(str(filename)) as rmb_data:
        time = np.asarray(rmb_data[DataKey.TIME][:]).reshape(-1)
        object_pose = np.asarray(rmb_data[DataKey.MEASURED_TBLOCK_POSE][:])
    displacement = np.linalg.norm(object_pose[:, :3] - object_pose[0, :3], axis=1)
    motion_idxes = np.flatnonzero(displacement >= 0.005)
    motion_idx = int(motion_idxes[0]) if len(motion_idxes) > 0 else None
    return time, motion_idx


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
        "motion_mean_shift": np.mean(
            [abs(record["mean_at_motion"] - record["initial_pb"]) for record in records]
        ),
        "motion_std_ratio": np.mean(
            [record["std_at_motion"] / record["initial_std"] for record in records]
        ),
    }


def save_csv(path, records):
    fieldnames = [
        "beta",
        "object_id",
        "episode",
        "target_pb",
        "initial_pb",
        "initial_std",
        "num_updates",
        "final_mean",
        "final_std",
        "final_error",
        "covered_by_2std",
        "motion_time_s",
        "mean_at_motion",
        "std_at_motion",
    ]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record[key] for key in fieldnames})


def plot_selected_trajectories(path, selected_records, selected_beta, num_points):
    fig, axes = plt.subplots(3, 5, figsize=(18, 9), sharex=False, sharey=True)
    for object_id in range(3):
        records = sorted(
            [r for r in selected_records if r["object_id"] == object_id],
            key=lambda record: record["episode"],
        )
        assert len(records) == 5, (object_id, len(records))
        for episode_idx, (ax, record) in enumerate(zip(axes[object_id], records)):
            time = record["time"]
            mean = record["mean_trajectory"]
            std = record["std_trajectory"]
            ax.fill_between(
                time,
                mean - 2.0 * std,
                mean + 2.0 * std,
                color="#4c78a8",
                alpha=0.2,
                label="mean ± 2 std" if object_id == 0 and episode_idx == 0 else None,
            )
            ax.plot(
                time,
                mean,
                color="#1f5a94",
                linewidth=1.8,
                label="belief mean" if object_id == 0 and episode_idx == 0 else None,
            )
            ax.axhline(
                record["target_pb"],
                color="#d62728",
                linestyle="--",
                linewidth=1.3,
                label="learned target PB"
                if object_id == 0 and episode_idx == 0
                else None,
            )
            if np.isfinite(record["motion_time_s"]):
                ax.axvline(
                    record["motion_time_s"],
                    color="#2ca02c",
                    linestyle=":",
                    linewidth=1.2,
                    label="5 mm motion"
                    if object_id == 0 and episode_idx == 0
                    else None,
                )
            ax.set_title(record["episode"].replace("WrenchPredObject", "O"), fontsize=9)
            ax.grid(alpha=0.22)
            if episode_idx == 0:
                ax.set_ylabel(f"Object{object_id}\nPB")
            if object_id == 2:
                ax.set_xlabel("time [s]")
    axes[0, 0].legend(loc="lower left", fontsize=7)
    fig.suptitle(
        f"Gaussian-belief Online PB: beta={selected_beta:g}, "
        f"{num_points}-point Gauss-Hermite, every overlapping window"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_beta_sweep(path, grouped_summary, beta_list):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    colors = ["#4c78a8", "#f58518", "#54a24b"]
    for object_id, color in zip(range(3), colors):
        summaries = [grouped_summary[(beta, object_id)] for beta in beta_list]
        axes[0].plot(
            beta_list,
            [summary["rmse"] for summary in summaries],
            marker="o",
            color=color,
            label=f"Object{object_id}",
        )
        axes[1].plot(
            beta_list,
            [summary["belief_std"] for summary in summaries],
            marker="o",
            color=color,
        )
        axes[2].plot(
            beta_list,
            [summary["coverage_2std"] for summary in summaries],
            marker="o",
            color=color,
        )
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("beta")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("final PB RMSE")
    axes[1].set_ylabel("mean final belief std")
    axes[2].set_ylabel("final ±2 std coverage")
    axes[2].set_ylim(-0.05, 1.05)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def format_summary_table(grouped_summary, beta_list):
    rows = []
    for beta in beta_list:
        for object_id in range(3):
            summary = grouped_summary[(beta, object_id)]
            rows.append(
                "<tr>"
                f"<td>{beta:g}</td><td>Object{object_id}</td>"
                f"<td>{summary['final_mean']:.5f} ± "
                f"{summary['cross_episode_std']:.5f}</td>"
                f"<td>{summary['rmse']:.5f}</td>"
                f"<td>{summary['belief_std']:.5f}</td>"
                f"<td>{summary['coverage_2std'] * 100:.0f}%</td>"
                f"<td>{summary['motion_mean_shift']:.5f}</td>"
                f"<td>{summary['motion_std_ratio'] * 100:.1f}%</td>"
                "</tr>"
            )
    return "\n".join(rows)


def write_html(path, args, initial_pb, target_pb, grouped_summary):
    selected_rows = []
    for object_id in range(3):
        summary = grouped_summary[(args.selected_beta, object_id)]
        selected_rows.append(
            "<tr>"
            f"<td>Object{object_id}</td><td>{target_pb[object_id]:.5f}</td>"
            f"<td>{summary['final_mean']:.5f} ± "
            f"{summary['cross_episode_std']:.5f}</td>"
            f"<td>{summary['rmse']:.5f}</td>"
            f"<td>{summary['belief_std']:.5f}</td>"
            f"<td>{summary['coverage_2std'] * 100:.0f}%</td>"
            "</tr>"
        )
    source_checkpoint = html.escape(str(args.checkpoint.resolve()))
    source_dataset = html.escape(str(args.dataset_dir.resolve()))
    beta_table = format_summary_table(grouped_summary, args.beta_list)
    path.write_text(
        f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Gaussian-belief Online PB validation</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:1180px;margin:32px auto;padding:0 20px;line-height:1.65;color:#20242a}}
h1,h2{{line-height:1.25}} code{{background:#f1f3f5;padding:.12em .35em;border-radius:4px}}
.note{{background:#eef6ff;border-left:4px solid #4c78a8;padding:12px 16px}}
table{{border-collapse:collapse;width:100%;font-size:.92rem}} th,td{{border:1px solid #ccd2d8;padding:7px 9px;text-align:right}}
th:first-child,td:first-child{{text-align:left}} img{{width:100%;height:auto;border:1px solid #d8dde3}}
.small{{color:#5a6470;font-size:.9rem}}
</style></head><body>
<h1>Gaussian-belief Online PB — state-based validation</h1>
<p class="note"><strong>結果の読み方:</strong> 青線は Online PB の平均 μ、青帯は μ±2σ、赤破線は各 object の learned PB、緑点線は物体が初期位置から5 mm動いた時刻です。これは recorded validation replay であり、closed-loop rollout の保証ではありません。</p>
<h2>条件</h2>
<ul>
<li>checkpoint: <code>{source_checkpoint}</code>（policy_best）</li>
<li>validation: <code>{source_dataset}</code>（Object0/1/2 各5 episode）</li>
<li>初期 belief: N({initial_pb:.6f}, {args.initial_std:.3f}²)、initial object ID={args.initial_object_id}</li>
<li>Gauss–Hermite: M={args.num_points}、pose prediction lossのみ、従来backprop版と同じ全overlapping window更新</li>
<li>表示用設定: β={args.selected_beta:g}。β は validation で比較中の pseudo-likelihood scale で、確率校正済みの観測 noise ではありません。</li>
</ul>
<h2>15 episode の推定曲線</h2>
<img src="selected_beta_trajectories.png" alt="15 validation PB belief trajectories">
<h2>β={args.selected_beta:g} の終端集計</h2>
<table><thead><tr><th>object</th><th>target PB</th><th>final μ mean ± episode std</th><th>RMSE</th><th>mean final σ</th><th>target in μ±2σ</th></tr></thead><tbody>
{''.join(selected_rows)}
</tbody></table>
<p class="small">final μ の episode std は episode 間の再現性、final σ は各 episode 内の belief 幅です。異なる量なので両方を分けて表示しています。</p>
<h2>β sweep</h2>
<img src="beta_sweep.png" alt="beta sweep metrics">
<table><thead><tr><th>β</th><th>object</th><th>final μ mean ± episode std</th><th>RMSE</th><th>mean final σ</th><th>coverage</th><th>|μ_motion−μ0|</th><th>σ_motion/σ0</th></tr></thead><tbody>
{beta_table}
</tbody></table>
<h2>更新式</h2>
<p><code>p_i = μ + σ z_i</code> を1 batchで WP4 に入れ、<code>w_i = softmax(log a_i − β L_i)</code>、<code>μ' = Σw_i p_i</code>、<code>σ'² = Σw_i(p_i−μ')²</code> と更新しました。loss が全候補で同じなら μ と σ は変わりません。</p>
<h2>判断上の注意</h2>
<p>βを大きくすると target へ速く寄る場合がありますが、同時に σ が急速に潰れます。したがって final mean の近さだけでは選びません。5 mm motion 前の平均移動・収縮、±2σ coverage、次 window の予測誤差、最終的には同じ条件の closed-loop rollout を併せて判断します。</p>
<p class="small">Raw numeric records: <a href="episode_summary.csv">episode_summary.csv</a></p>
</body></html>""",
        encoding="utf-8",
    )


def main():
    args = parse_args()
    assert torch.cuda.is_available(), "This validation replay must run on GPU."
    assert args.selected_beta in args.beta_list, (args.selected_beta, args.beta_list)
    filenames = sorted(args.dataset_dir.glob("WrenchPredObject[0-2]/*.rmb"))
    assert len(filenames) == 15, len(filenames)
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
    target_pb = pb_table[:3, 0]

    records = []
    selected_records = []
    for beta in args.beta_list:
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
                    beta=beta,
                )
            )
            mean_trajectory = mean_trajectory[:, 0]
            std_trajectory = std_trajectory[:, 0]
            if motion_idx is None:
                motion_time = np.nan
                mean_at_motion = mean_trajectory[-1]
                std_at_motion = std_trajectory[-1]
            else:
                motion_time = time[motion_idx]
                mean_at_motion = mean_trajectory[motion_idx]
                std_at_motion = std_trajectory[motion_idx]
            record = {
                "beta": beta,
                "object_id": object_id,
                "episode": filename.stem,
                "target_pb": float(target_pb[object_id]),
                "initial_pb": float(initial_pb[0]),
                "initial_std": args.initial_std,
                "num_updates": num_updates,
                "final_mean": float(final_mean[0]),
                "final_std": float(final_std[0]),
                "final_error": float(final_mean[0] - target_pb[object_id]),
                "covered_by_2std": bool(
                    abs(final_mean[0] - target_pb[object_id]) <= 2.0 * final_std[0]
                ),
                "motion_time_s": float(motion_time),
                "mean_at_motion": float(mean_at_motion),
                "std_at_motion": float(std_at_motion),
            }
            records.append(record)
            if beta == args.selected_beta:
                selected_record = record.copy()
                selected_record.update(
                    {
                        "time": time,
                        "mean_trajectory": mean_trajectory,
                        "std_trajectory": std_trajectory,
                    }
                )
                selected_records.append(selected_record)
            print(
                f"[{episode_idx:02d}/15] beta={beta:g} {filename.stem}: "
                f"mean={final_mean[0]:.5f}, std={final_std[0]:.5f}"
            )

    grouped_records = defaultdict(list)
    for record in records:
        grouped_records[(record["beta"], record["object_id"])].append(record)
    grouped_summary = {
        key: summarize(group_records) for key, group_records in grouped_records.items()
    }

    save_csv(args.output_dir / "episode_summary.csv", records)
    plot_selected_trajectories(
        args.output_dir / "selected_beta_trajectories.png",
        selected_records,
        args.selected_beta,
        args.num_points,
    )
    plot_beta_sweep(
        args.output_dir / "beta_sweep.png",
        grouped_summary,
        args.beta_list,
    )
    write_html(
        args.output_dir / "gaussian_belief_validation.html",
        args,
        float(initial_pb[0]),
        target_pb,
        grouped_summary,
    )
    print(f"report: {args.output_dir / 'gaussian_belief_validation.html'}")


if __name__ == "__main__":
    main()
