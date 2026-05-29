import argparse
import csv
from pathlib import Path

import numpy as np

from robo_manip_baselines.common import DataKey, RmbData

RL_POLICY_STATE_KEY = "rl_policy_state"

STATE_COMPONENTS = [
    ("agent/qpos", 8),
    ("agent/qvel", 7),
    ("extra/inner_marker_panel_position", 3),
    ("extra/inner_marker_panel_rotation_6d", 6),
    ("extra/outer_marker_panel_position", 3),
    ("extra/outer_marker_panel_rotation_6d", 6),
]


def parse_argument():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("rmb_path", type=Path, help="path to data (*.rmb or *.hdf5)")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="output CSV path",
    )
    parser.add_argument(
        "--state-key",
        type=str,
        default=RL_POLICY_STATE_KEY,
        help="RMB dataset key containing the 33D RL policy state",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite output CSV if it already exists",
    )
    return parser.parse_args()


def default_output_path(rmb_path: Path) -> Path:
    path = Path(str(rmb_path).rstrip("/"))
    if path.suffix in (".rmb", ".hdf5"):
        return path.with_suffix(".state.csv")
    return path.with_name(path.name + ".state.csv")


def state_headers():
    headers = []
    for name, dim in STATE_COMPONENTS:
        safe_name = name.replace("/", "_")
        headers.extend([f"state_{safe_name}_{idx}" for idx in range(dim)])
    return headers


def load_state(rmb_data: RmbData, state_key: str) -> np.ndarray:
    if state_key not in rmb_data.keys():
        available = ", ".join(sorted(rmb_data.keys()))
        raise KeyError(f"'{state_key}' is missing. Available keys: {available}")

    state = np.asarray(rmb_data[state_key][:])
    if state.ndim != 2:
        raise ValueError(f"'{state_key}' must be 2D, got shape {state.shape}.")
    expected_dim = sum(dim for _, dim in STATE_COMPONENTS)
    if state.shape[1] != expected_dim:
        raise ValueError(
            f"Unexpected state dim: {state.shape[1]}, expected {expected_dim}."
        )
    return state


def run(rmb_path: Path, output: Path | None, state_key: str, force: bool):
    output = default_output_path(rmb_path) if output is None else output
    if output.exists() and not force:
        raise FileExistsError(f"Output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    with RmbData(str(rmb_path)) as rmb_data:
        state = load_state(rmb_data, state_key)
        if DataKey.TIME in rmb_data.keys():
            time = np.asarray(rmb_data[DataKey.TIME][:]).reshape(-1)
        else:
            time = np.arange(len(state), dtype=np.float64)

        row_count = min(len(state), len(time))
        if len(state) != len(time):
            print(
                f"[ExportRLPolicyStateCsv] Length mismatch state={len(state)}, "
                f"time={len(time)}; export first {row_count} rows."
            )

        with open(output, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestep", "time"] + state_headers())
            for idx in range(row_count):
                writer.writerow(
                    [idx, float(time[idx])] + state[idx].astype(float).tolist()
                )

    print(f"[ExportRLPolicyStateCsv] Saved: {output}")


if __name__ == "__main__":
    run(**vars(parse_argument()))
