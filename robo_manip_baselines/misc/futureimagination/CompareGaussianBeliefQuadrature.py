import argparse
import csv
import html
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
        description="Compare Gaussian-belief Online PB quadrature orders."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--num_points", type=int, nargs=2, default=[9, 128])
    parser.add_argument("--initial_object_id", type=int, default=0)
    parser.add_argument("--initial_std", type=float, default=0.25)
    parser.add_argument("--beta", type=float, default=10.0)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    return parser.parse_args()


def replay(filename, num_points, args, initial_pb, policy, model_meta_info, device):
    mean, std, num_updates, final_mean, final_std = adapt_pb_trajectory(
        str(filename),
        initial_pb,
        policy,
        model_meta_info,
        device,
        learning_rate=0.006,
        wrench_loss_weight=0.0,
        update_type="gaussian_belief",
        initial_std=args.initial_std,
        num_points=num_points,
        beta=args.beta,
    )
    with RmbData(str(filename)) as rmb_data:
        time = np.asarray(rmb_data[DataKey.TIME][:]).reshape(-1)
    return {
        "time": time,
        "mean": mean[:, 0],
        "std": std[:, 0],
        "num_updates": num_updates,
        "final_mean": float(final_mean[0]),
        "final_std": float(final_std[0]),
    }


def summarize(records, order, object_id, target):
    selected = [
        record
        for record in records
        if record["order"] == order and record["object_id"] == object_id
    ]
    final_mean = np.asarray([record["final_mean"] for record in selected])
    final_std = np.asarray([record["final_std"] for record in selected])
    return {
        "mean": float(final_mean.mean()),
        "episode_std": float(final_mean.std()),
        "rmse": float(np.sqrt(np.mean(np.square(final_mean - target)))),
        "belief_std": float(final_std.mean()),
        "coverage": float(np.mean(np.abs(final_mean - target) <= 2.0 * final_std)),
    }


def plot_trajectories(path, paired_records, orders, target_pb, beta):
    fig, axes = plt.subplots(3, 5, figsize=(18, 9), sharey=True)
    colors = ["#2563a6", "#e07a1f"]
    for pair, ax in zip(paired_records, axes.flat):
        object_id = pair["object_id"]
        for order, color in zip(orders, colors):
            result = pair[order]
            ax.plot(
                result["time"],
                result["mean"],
                color=color,
                linewidth=1.7,
                label=f"M={order}" if pair["episode_index"] == 0 else None,
            )
            ax.fill_between(
                result["time"],
                result["mean"] - 2.0 * result["std"],
                result["mean"] + 2.0 * result["std"],
                color=color,
                alpha=0.09,
            )
        ax.axhline(target_pb[object_id], color="#c43b35", linestyle="--", linewidth=1.2)
        ax.set_title(pair["episode"].replace("WrenchPredObject", "O"), fontsize=9)
        ax.grid(alpha=0.2)
        if pair["episode_index"] == 0:
            ax.set_ylabel(f"Object{object_id}\nPB")
        if object_id == 2:
            ax.set_xlabel("time [s]")
    axes[0, 0].legend(loc="best", fontsize=8)
    fig.suptitle(
        f"Gaussian-belief Online PB: {orders[0]} vs {orders[1]} points, beta={beta:g}\n"
        "line = posterior mean, pale band = mean +/- 2 std"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_csv(path, paired_records, orders, target_pb):
    fields = [
        "object_id",
        "episode",
        "target_pb",
        *[
            field
            for order in orders
            for field in (f"final_mean_m{order}", f"final_std_m{order}")
        ],
        "final_mean_difference",
        "final_std_difference",
        "max_trajectory_mean_difference",
        "max_trajectory_std_difference",
    ]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for pair in paired_records:
            first = pair[orders[0]]
            second = pair[orders[1]]
            writer.writerow(
                {
                    "object_id": pair["object_id"],
                    "episode": pair["episode"],
                    "target_pb": target_pb[pair["object_id"]],
                    f"final_mean_m{orders[0]}": first["final_mean"],
                    f"final_std_m{orders[0]}": first["final_std"],
                    f"final_mean_m{orders[1]}": second["final_mean"],
                    f"final_std_m{orders[1]}": second["final_std"],
                    "final_mean_difference": second["final_mean"] - first["final_mean"],
                    "final_std_difference": second["final_std"] - first["final_std"],
                    "max_trajectory_mean_difference": np.max(
                        np.abs(second["mean"] - first["mean"])
                    ),
                    "max_trajectory_std_difference": np.max(
                        np.abs(second["std"] - first["std"])
                    ),
                }
            )


def write_html(path, args, orders, target_pb, summaries, paired_records):
    table_rows = []
    for object_id in range(3):
        for order in orders:
            summary = summaries[(order, object_id)]
            table_rows.append(
                "<tr>"
                f"<td>Object{object_id}</td><td>{order}</td>"
                f"<td>{target_pb[object_id]:.5f}</td>"
                f"<td>{summary['mean']:.5f} &plusmn; {summary['episode_std']:.5f}</td>"
                f"<td>{summary['rmse']:.5f}</td><td>{summary['belief_std']:.5f}</td>"
                f"<td>{summary['coverage'] * 100:.0f}%</td></tr>"
            )
    mean_differences = np.asarray(
        [
            pair[orders[1]]["final_mean"] - pair[orders[0]]["final_mean"]
            for pair in paired_records
        ]
    )
    std_differences = np.asarray(
        [
            pair[orders[1]]["final_std"] - pair[orders[0]]["final_std"]
            for pair in paired_records
        ]
    )
    max_curve_mean_difference = max(
        float(np.max(np.abs(pair[orders[1]]["mean"] - pair[orders[0]]["mean"])))
        for pair in paired_records
    )
    max_curve_std_difference = max(
        float(np.max(np.abs(pair[orders[1]]["std"] - pair[orders[0]]["std"])))
        for pair in paired_records
    )
    checkpoint = html.escape(str(args.checkpoint.resolve()))
    dataset = html.escape(str(args.dataset_dir.resolve()))
    path.write_text(
        f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Online PB: {orders[0]} vs {orders[1]} candidates</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:1180px;margin:32px auto;padding:0 20px;line-height:1.7;color:#20242a}}
h1,h2{{line-height:1.3}} code{{background:#f0f2f4;padding:.12em .35em;border-radius:4px}}
.note{{background:#eef6ff;border-left:4px solid #2563a6;padding:12px 16px}}
table{{border-collapse:collapse;width:100%;font-size:.92rem}}th,td{{border:1px solid #ccd2d8;padding:7px 9px;text-align:right}}
th:first-child,td:first-child{{text-align:left}}img{{width:100%;border:1px solid #d8dde3}}.small{{color:#5c6670;font-size:.9rem}}
</style></head><body>
<h1>Gaussian Online PB: {orders[0]}候補 vs {orders[1]}候補</h1>
<p class="note">候補数だけを変え、同じvalidation 15本を再生した比較です。{orders[1]}候補は別の粒子を保持するparticle filterではなく、各windowで同じGaussianをより細かく数値積分します。</p>
<h2>条件</h2><ul>
<li>state-based WP4: <code>{checkpoint}</code>（policy_best）</li>
<li>validation: <code>{dataset}</code>（Object0/1/2 各5本）</li>
<li>初期belief: N(0.199588, {args.initial_std:.3f}&sup2;), beta={args.beta:g}, pose lossのみ</li>
<li>比較: Gauss-Hermite M={orders[0]} と M={orders[1]}、全overlapping window更新</li>
</ul>
<h2>結論を読むための差分</h2>
<ul>
<li>終端mean差（M={orders[1]} minus M={orders[0]}）: 平均 {mean_differences.mean():+.6f}、最大絶対値 {np.max(np.abs(mean_differences)):.6f}</li>
<li>終端std差: 平均 {std_differences.mean():+.6f}、最大絶対値 {np.max(np.abs(std_differences)):.6f}</li>
<li>全時刻での最大 |mean差|: {max_curve_mean_difference:.6f}</li>
<li>全時刻での最大 |std差|: {max_curve_std_difference:.6f}</li>
</ul>
<p>M={orders[1]}で改善が小さければ、M={orders[0]}はこの1次元Gaussian積分には十分です。差が大きい場合は、9点近似の粗さだけでなく、128点側が学習PB範囲から遠い候補までWP4へ入れる外挿の影響も疑う必要があります。</p>
<h2>15本の曲線</h2><img src="quadrature_comparison.png" alt="quadrature comparison trajectories">
<h2>object別の終端値</h2>
<table><thead><tr><th>object</th><th>M</th><th>target PB</th><th>final mean &plusmn; episode std</th><th>RMSE</th><th>mean final std</th><th>target in mean&plusmn;2std</th></tr></thead><tbody>
{''.join(table_rows)}
</tbody></table>
<h2>更新処理</h2>
<p><code>p_i = mean + std * z_i</code> をまとめてWP4へ通し、<code>w_i = softmax(log(a_i) - beta * loss_i)</code>、重み付き1次・2次モーメントから次のmean/stdを得ます。候補数が増えるのはこの積分近似の解像度であり、観測情報そのものは増えません。</p>
<p class="small">Raw episode comparison: <a href="quadrature_comparison.csv">quadrature_comparison.csv</a></p>
</body></html>""",
        encoding="utf-8",
    )


def main():
    args = parse_args()
    orders = tuple(args.num_points)
    assert len(orders) == 2 and min(orders) >= 3 and orders[0] != orders[1], orders
    if args.device == "cuda":
        assert torch.cuda.is_available()
    device = torch.device(args.device)
    filenames = sorted(args.dataset_dir.glob("WrenchPredObject[0-2]/*.rmb"))
    assert len(filenames) == 15, len(filenames)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model_meta_info = load_model_meta_info(args.checkpoint)
    policy = load_policy(args.checkpoint, model_meta_info, device)
    initial_pb, _ = load_pb(args.checkpoint, args.initial_object_id, model_meta_info)
    pb_table, _ = load_pb_table(args.checkpoint, model_meta_info)
    target_pb = pb_table[:3, 0]

    records = []
    paired_records = []
    object_episode_index = {object_id: 0 for object_id in range(3)}
    for file_index, filename in enumerate(filenames, start=1):
        object_id = int(filename.parent.name.removeprefix("WrenchPredObject"))
        pair = {
            "object_id": object_id,
            "episode": filename.stem,
            "episode_index": object_episode_index[object_id],
        }
        object_episode_index[object_id] += 1
        for order in orders:
            result = replay(
                filename, order, args, initial_pb, policy, model_meta_info, device
            )
            pair[order] = result
            records.append({"order": order, "object_id": object_id, **result})
            print(
                f"[{file_index:02d}/15] M={order} {filename.stem}: "
                f"mean={result['final_mean']:.6f}, std={result['final_std']:.6f}"
            )
        paired_records.append(pair)

    summaries = {
        (order, object_id): summarize(
            records, order, object_id, float(target_pb[object_id])
        )
        for order in orders
        for object_id in range(3)
    }
    save_csv(
        args.output_dir / "quadrature_comparison.csv", paired_records, orders, target_pb
    )
    plot_trajectories(
        args.output_dir / "quadrature_comparison.png",
        paired_records,
        orders,
        target_pb,
        args.beta,
    )
    write_html(
        args.output_dir / "gaussian_belief_9_vs_128_validation.html",
        args,
        orders,
        target_pb,
        summaries,
        paired_records,
    )
    print("report: " f"{args.output_dir / 'gaussian_belief_9_vs_128_validation.html'}")


if __name__ == "__main__":
    main()
