import argparse
import re


EPISODE_HEADER_PATTERN = re.compile(r"\[EvalWrenchPredictorDir\] (.+\.rmb) MAE:")
METRIC_PATTERN = re.compile(
    r"  - (?P<name>Fx/Fy mean|torque mean|Fz only|Fx|Fy|Fz|Nx|Ny|Nz): (?P<value>[-+0-9.eE]+)"
)


def parse_argument():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("log_a", type=str, help="first EvalWrenchPredictorDir log")
    parser.add_argument("log_b", type=str, help="second EvalWrenchPredictorDir log")
    return parser.parse_args()


def parse_eval_log(log_path):
    episode_to_metrics = {}
    current_episode = None

    with open(log_path, "r") as f:
        for line in f:
            line = line.rstrip("\n")

            episode_match = EPISODE_HEADER_PATTERN.match(line)
            if episode_match is not None:
                current_episode = episode_match.group(1)
                episode_to_metrics[current_episode] = {}
                continue

            if current_episode is None:
                continue

            metric_match = METRIC_PATTERN.match(line)
            if metric_match is None:
                continue

            metric_name = metric_match.group("name")
            metric_value = float(metric_match.group("value"))
            episode_to_metrics[current_episode][metric_name] = metric_value

    assert len(episode_to_metrics) > 0, log_path
    return episode_to_metrics


def require_metric(metrics, episode, metric_name, log_path):
    assert metric_name in metrics, f"{metric_name} is missing for {episode}: {log_path}"
    return metrics[metric_name]


def main():
    args = parse_argument()
    log_a = parse_eval_log(args.log_a)
    log_b = parse_eval_log(args.log_b)

    common_episodes = sorted(set(log_a) & set(log_b))
    assert len(common_episodes) > 0, "No common RMB episodes are found."

    episode_diff_list = []
    for episode in common_episodes:
        force_xy_a = require_metric(
            log_a[episode], episode, "Fx/Fy mean", args.log_a
        )
        force_xy_b = require_metric(
            log_b[episode], episode, "Fx/Fy mean", args.log_b
        )
        torque_a = require_metric(log_a[episode], episode, "torque mean", args.log_a)
        torque_b = require_metric(log_b[episode], episode, "torque mean", args.log_b)

        force_xy_diff = abs(force_xy_a - force_xy_b)
        torque_diff = abs(torque_a - torque_b)
        combined_diff = force_xy_diff + torque_diff
        episode_diff_list.append(
            {
                "episode": episode,
                "combined_diff": combined_diff,
                "force_xy_diff": force_xy_diff,
                "torque_diff": torque_diff,
                "force_xy_a": force_xy_a,
                "force_xy_b": force_xy_b,
                "torque_a": torque_a,
                "torque_b": torque_b,
            }
        )

    max_combined = max(episode_diff_list, key=lambda item: item["combined_diff"])
    max_force_xy = max(episode_diff_list, key=lambda item: item["force_xy_diff"])
    max_torque = max(episode_diff_list, key=lambda item: item["torque_diff"])

    print(f"Compared episodes: {len(common_episodes)}")
    print()
    print_result("Max combined difference", max_combined, args.log_a, args.log_b)
    print()
    print_result("Max Fx/Fy mean difference", max_force_xy, args.log_a, args.log_b)
    print()
    print_result("Max torque mean difference", max_torque, args.log_a, args.log_b)


def lower_log_name(value_a, value_b, log_a_path, log_b_path):
    if value_a < value_b:
        return log_a_path
    if value_b < value_a:
        return log_b_path
    return "tie"


def print_result(title, result, log_a_path, log_b_path):
    print(f"[{title}]")
    print(f"  - episode: {result['episode']}")
    print(f"  - combined diff: {result['combined_diff']:.6f}")
    print(
        f"  - Fx/Fy mean: {result['force_xy_a']:.6f} vs {result['force_xy_b']:.6f} "
        f"(diff {result['force_xy_diff']:.6f})"
    )
    print(
        "    lower MAE: "
        f"{lower_log_name(result['force_xy_a'], result['force_xy_b'], log_a_path, log_b_path)}"
    )
    print(
        f"  - torque mean: {result['torque_a']:.6f} vs {result['torque_b']:.6f} "
        f"(diff {result['torque_diff']:.6f})"
    )
    print(
        "    lower MAE: "
        f"{lower_log_name(result['torque_a'], result['torque_b'], log_a_path, log_b_path)}"
    )


if __name__ == "__main__":
    main()
