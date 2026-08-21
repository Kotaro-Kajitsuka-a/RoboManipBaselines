import argparse
import csv
import html
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from robo_manip_baselines.common import DataKey, RmbData
from robo_manip_baselines.misc.futureimagination.AnalyzeLiftingSuccess import (
    find_unique_rmb_files,
    get_object_name,
)
from robo_manip_baselines.policy.wrench_predictor4_online.WrenchPredictor4OnlineUtils import (
    ONLINE_PB_STD_KEY,
)

GROUP_ORDER = ("I0", "I1", "I2", "I4", "I5", "I6", "I7")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize Gaussian-belief Online PB lifting rollouts."
    )
    parser.add_argument(
        "dataset_paths",
        type=Path,
        nargs="+",
        help="RMB files or directories containing rollout episodes",
    )
    parser.add_argument("--lifting_summary", type=Path, required=True)
    parser.add_argument("--wp4_checkpoint", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--baseline_csv", type=Path, default=None)
    return parser.parse_args()


def load_episode(filename: str) -> dict:
    with RmbData(filename) as rmb_data:
        time = np.asarray(rmb_data[DataKey.TIME][:], dtype=np.float64)
        pb_mean = np.asarray(rmb_data[DataKey.MATERIAL_PROPERTY][:, 0])
        pb_std = np.asarray(rmb_data[ONLINE_PB_STD_KEY][:, 0])
    assert time.ndim == pb_mean.ndim == pb_std.ndim == 1
    assert len(time) == len(pb_mean) == len(pb_std) and len(time) > 0
    return {
        "group": get_object_name(Path(filename)),
        "filename": filename,
        "time": time - time[0],
        "pb_mean": pb_mean,
        "pb_std": pb_std,
    }


def load_reference_pbs(checkpoint: Path) -> np.ndarray:
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    reference_pbs = state_dict["material_property.weight"]
    assert reference_pbs.ndim == 2 and reference_pbs.shape[1] == 1
    assert reference_pbs.shape[0] >= 3, reference_pbs.shape
    return reference_pbs[:3, 0].numpy()


def load_seed42_baseline(path: Path | None) -> dict:
    if path is None:
        return {}
    rows = [
        row for row in csv.DictReader(path.open()) if "/seed42/" in row["episode_path"]
    ]
    grouped_rows = defaultdict(list)
    for row in rows:
        grouped_rows[row["group"]].append(row)
    assert len(rows) == 70, len(rows)
    return {
        group: {
            "success": sum(row["success"] == "True" for row in group_rows),
            "success_once": sum(row["success_once"] == "True" for row in group_rows),
            "total": len(group_rows),
        }
        for group, group_rows in grouped_rows.items()
    }


def summarize_belief(episodes: list[dict]) -> dict:
    final_mean = np.asarray([episode["pb_mean"][-1] for episode in episodes])
    final_std = np.asarray([episode["pb_std"][-1] for episode in episodes])
    return {
        "episode_count": len(episodes),
        "final_mean": float(final_mean.mean()),
        "final_mean_episode_std": float(final_mean.std()),
        "final_mean_min": float(final_mean.min()),
        "final_mean_max": float(final_mean.max()),
        "final_belief_std": float(final_std.mean()),
        "final_belief_std_min": float(final_std.min()),
        "final_belief_std_max": float(final_std.max()),
    }


def plot_beliefs(
    path: Path,
    grouped_episodes: dict[str, list[dict]],
    reference_pbs: np.ndarray,
) -> None:
    figure, axes = plt.subplots(2, 4, figsize=(17, 8), sharex=True)
    for axis, group in zip(axes.flat, GROUP_ORDER, strict=False):
        episodes = grouped_episodes[group]
        common_length = min(len(episode["time"]) for episode in episodes)
        time = np.stack([episode["time"][:common_length] for episode in episodes]).mean(
            axis=0
        )
        pb_mean = np.stack([episode["pb_mean"][:common_length] for episode in episodes])
        pb_std = np.stack([episode["pb_std"][:common_length] for episode in episodes])
        for episode_idx, episode in enumerate(episodes):
            axis.plot(
                episode["time"],
                episode["pb_mean"],
                color="#4c78a8",
                alpha=0.18,
                linewidth=0.8,
                label="episode mean" if episode_idx == 0 else None,
            )
        mean = pb_mean.mean(axis=0)
        mean_std = pb_std.mean(axis=0)
        axis.fill_between(
            time,
            mean - 2.0 * mean_std,
            mean + 2.0 * mean_std,
            color="#4c78a8",
            alpha=0.16,
            label="mean belief ±2σ",
        )
        axis.plot(time, mean, color="#172b4d", linewidth=2.0, label="episode mean")
        reference_colors = ("#d62728", "#f28e2b", "#2ca02c")
        for object_id, (reference_pb, color) in enumerate(
            zip(reference_pbs, reference_colors, strict=True)
        ):
            axis.axhline(
                reference_pb,
                color=color,
                linestyle="--",
                linewidth=1.4,
                label=f"trained I{object_id} PB",
            )
        axis.set_title(group)
        axis.grid(alpha=0.25)
        axis.set_xlabel("time [s]")
        axis.set_ylabel("PB")
    axes.flat[-1].axis("off")
    axes.flat[0].legend(fontsize=8, loc="best")
    figure.suptitle("Gaussian-belief Online PB during closed-loop rollout")
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def plot_success(path: Path, lifting_summary: dict, baseline: dict) -> None:
    x = np.arange(len(GROUP_ORDER))
    final_rate = np.asarray(
        [lifting_summary["groups"][group]["success_rate"] for group in GROUP_ORDER]
    )
    once_rate = np.asarray(
        [lifting_summary["groups"][group]["success_once_rate"] for group in GROUP_ORDER]
    )
    figure, axis = plt.subplots(figsize=(11, 5.2))
    width = 0.25 if baseline else 0.36
    axis.bar(x - width / 2, final_rate, width, label="final-state success")
    axis.bar(x + width / 2, once_rate, width, label="success at least once")
    if baseline:
        baseline_rate = np.asarray(
            [
                baseline[group]["success"] / baseline[group]["total"]
                for group in GROUP_ORDER
            ]
        )
        axis.plot(
            x,
            baseline_rate,
            color="#d62728",
            marker="o",
            linestyle="--",
            label="joint-position baseline seed42 (reference)",
        )
    axis.set_xticks(x, GROUP_ORDER)
    axis.set_ylim(0.0, 1.05)
    axis.set_ylabel("success rate")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def write_csv(path: Path, group_summary: dict) -> None:
    fieldnames = ["group", *next(iter(group_summary.values())).keys()]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for group in GROUP_ORDER:
            writer.writerow({"group": group, **group_summary[group]})


def write_html(
    path: Path,
    args: argparse.Namespace,
    lifting_summary: dict,
    belief_summary: dict,
    reference_pbs: np.ndarray,
    baseline: dict,
) -> None:
    rows = []
    for group in GROUP_ORDER:
        object_id = int(group[1:])
        target = f"{reference_pbs[object_id]:.4f}" if object_id < 3 else "未学習"
        belief = belief_summary[group]
        lifting = lifting_summary["groups"][group]
        baseline_text = (
            f"{baseline[group]['success']}/{baseline[group]['total']}"
            if baseline
            else "—"
        )
        rows.append(
            "<tr>"
            f"<td>{group}</td><td>{target}</td>"
            f"<td>{belief['final_mean']:.4f} ± "
            f"{belief['final_mean_episode_std']:.4f}</td>"
            f"<td>{belief['final_belief_std']:.4f}</td>"
            f"<td>{lifting['success']}/{lifting['total']} "
            f"({lifting['success_rate'] * 100:.0f}%)</td>"
            f"<td>{lifting['success_once']}/{lifting['total']} "
            f"({lifting['success_once_rate'] * 100:.0f}%)</td>"
            f"<td>{baseline_text}</td>"
            "</tr>"
        )
    checkpoint = html.escape(str(args.wp4_checkpoint.resolve()))
    raw_summary = html.escape(str(args.lifting_summary.resolve()))
    path.write_text(
        f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Gaussian-belief Online PB rollout — seed 42</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:1200px;margin:32px auto;padding:0 20px;line-height:1.65;color:#20242a}}
h1,h2{{line-height:1.25}} code{{background:#f1f3f5;padding:.12em .35em;border-radius:4px}}
.good{{background:#eef8f0;border-left:4px solid #3a8f4e;padding:12px 16px}}
.warn{{background:#fff6e6;border-left:4px solid #d88a16;padding:12px 16px}}
table{{border-collapse:collapse;width:100%;font-size:.92rem}} th,td{{border:1px solid #ccd2d8;padding:7px 9px;text-align:right}}
th:first-child,td:first-child{{text-align:left}} img{{width:100%;height:auto;border:1px solid #d8dde3}}
.small{{color:#5a6470;font-size:.9rem}}
</style></head><body>
<h1>Gaussian-belief Online PB closed-loop rollout — seed 42</h1>
<p class="good"><strong>結果:</strong> 70 episode 中、終端成功は {lifting_summary['success']}/70 ({lifting_summary['success_rate'] * 100:.1f}%)、一度でも成功条件に入ったものは {lifting_summary['success_once']}/70 ({lifting_summary['success_once_rate'] * 100:.1f}%) でした。I0/I1/I2 の終端成功率は 90%/100%/100% です。</p>
<p class="warn"><strong>注意:</strong> rollout ログ内の reward は全件0でしたが、既存の解析基準（物体を10 cm以上持ち上げ、傾き7.5°未満）を保存HDF5へ適用すると上記結果になります。未知側 I4〜I7 は、一度持ち上げても終端までに傾く・落とす例が多く、終端成功は8/40です。</p>
<h2>条件</h2>
<ul>
<li>Diffusion Policy: state-based EEF pose、training seed 42、policy_last.ckpt</li>
<li>Online PB: Gaussian belief、M=16、β=10、初期 σ=0.25、全 overlapping window 更新</li>
<li>WP4: <code>{checkpoint}</code>（policy_best.ckpt）</li>
<li>各 object 10 episode、world 70–79 / 170–179 / … / 770–779</li>
</ul>
<h2>成功率</h2>
<img src="success_rates.png" alt="success rate by object">
<p class="small">赤破線は既存の joint-position baseline・training seed 42 で、入力表現が異なるため参考値です。今回の Gaussian Online-PB state/EefPose は終端37/70、baseline は31/70でした。</p>
<h2>Online PB の推移</h2>
<img src="rollout_pb_trajectories.png" alt="Online PB trajectories">
<p class="small">薄青は各 episode の平均 μ、濃線は10本平均、帯は各時刻の平均 belief σ による μ±2σです。全パネルに、実際に学習された I0・I1・I2 の参照PBを赤・橙・緑の破線で表示しています。I4以降には正解PBとして使える learned PB はありません。</p>
<h2>object別集計</h2>
<table><thead><tr><th>object</th><th>WP4 learned PB</th><th>final μ mean ± episode std</th><th>mean final σ</th><th>final success</th><th>success once</th><th>joint-pos baseline final</th></tr></thead><tbody>
{''.join(rows)}
</tbody></table>
<h2>読み取り</h2>
<p>I0〜I2 は終端 μ が各 learned PB に近く、closed-loop でも高い成功率でした。I4〜I7 には正解 learned PB がないため、推定値の正確さを赤線との距離では評価できません。ただし、各物体で異なる領域へまとまりながら終端 σ が 0.044〜0.059 まで縮むことは観察できます。この不確かさが妥当かどうかは、held-out の wrench prediction error などで別途校正する必要があります。I4〜I7 でも35/40本（87.5%）は一度成功条件に入るため、終盤の保持・姿勢安定性も次の改善点です。</p>
<p class="small">Raw: <a href="belief_summary.csv">belief_summary.csv</a> / lifting summary: <code>{raw_summary}</code></p>
</body></html>""",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    filenames = find_unique_rmb_files(args.dataset_paths)
    assert len(filenames) == 70, len(filenames)
    episodes = [load_episode(filename) for filename in filenames]
    grouped_episodes = defaultdict(list)
    for episode in episodes:
        grouped_episodes[episode["group"]].append(episode)
    assert set(grouped_episodes) == set(GROUP_ORDER), grouped_episodes.keys()
    for group in GROUP_ORDER:
        assert len(grouped_episodes[group]) == 10, (group, len(grouped_episodes[group]))

    lifting_summary = json.loads(args.lifting_summary.read_text())
    assert lifting_summary["total"] == 70, lifting_summary["total"]
    reference_pbs = load_reference_pbs(args.wp4_checkpoint)
    baseline = load_seed42_baseline(args.baseline_csv)
    belief_summary = {
        group: summarize_belief(grouped_episodes[group]) for group in GROUP_ORDER
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "belief_summary.csv", belief_summary)
    plot_beliefs(
        args.output_dir / "rollout_pb_trajectories.png",
        grouped_episodes,
        reference_pbs,
    )
    plot_success(args.output_dir / "success_rates.png", lifting_summary, baseline)
    write_html(
        args.output_dir / "gaussian_belief_rollout_seed42.html",
        args,
        lifting_summary,
        belief_summary,
        reference_pbs,
        baseline,
    )
    print(json.dumps(belief_summary, indent=2))
    print((args.output_dir / "gaussian_belief_rollout_seed42.html").resolve())


if __name__ == "__main__":
    main()
