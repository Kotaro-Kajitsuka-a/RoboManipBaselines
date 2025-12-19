#!/usr/bin/env python3
"""Run FT load identification on both xArm7s using the RealXarm7Dual env helpers.

Flow:
- Instantiate RealXarm7DualDemoEnv (reuses existing init_qpos and connection logic)
- Move both arms to the env's init pose via `_reset_robot` (internally uses `_set_action` with reset_bool=True and vel scale 0.1)
- For each arm: enable FT sensor, run iden_ft_sensor_load_offset, apply via set_ft_sensor_load_offset, then save_conf so it persists
"""

from __future__ import annotations

import argparse
import sys
import time

from robo_manip_baselines.envs.real.xarm7_dual.RealXarm7DualDemoEnv import (
    RealXarm7DualDemoEnv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Identify and save FT gravity compensation on both xArm7s (dual) using env helpers."
    )
    parser.add_argument("--ip-left", required=True, help="Left arm IP address (workspace perspective).")
    parser.add_argument("--ip-right", required=True, help="Right arm IP address (workspace perspective).")
    return parser.parse_args()


def identify_apply_save(arm_api, label: str) -> None:
    print(f"{label} enabling FT sensor and running load identification...")
    arm_api.set_ft_sensor_enable(1)
    start = time.monotonic()
    code, result = arm_api.iden_ft_sensor_load_offset()
    elapsed = time.monotonic() - start
    if code != 0:
        raise RuntimeError(f"{label} iden_ft_sensor_load_offset failed: code={code}")
    print(f"{label} identified offsets in {elapsed:.2f}s: {result}")

    apply_code = arm_api.set_ft_sensor_load_offset(result)
    if apply_code != 0:
        raise RuntimeError(f"{label} set_ft_sensor_load_offset failed: code={apply_code}")
    print(f"{label} applied load offsets")

    save_code = arm_api.save_conf()
    if save_code != 0:
        raise RuntimeError(f"{label} save_conf failed: code={save_code}")
    print(f"{label} saved configuration to controller flash")

    arm_api.set_ft_sensor_mode(0)
    arm_api.set_ft_sensor_enable(0)


def main() -> None:
    args = parse_args()

    # Use the demo env to reuse init_qpos and connection logic; cameras/gelsight are skipped with None.
    env = RealXarm7DualDemoEnv(
        robot_ip_left=args.ip_left,
        robot_ip_right=args.ip_right,
        camera_ids=None,
        gelsight_ids=None,
    )

    try:
        # Move both arms to init pose using existing reset logic (vel limit scale=0.1 inside).
        env._reset_robot()

        # Identify, apply, and save on each arm.
        identify_apply_save(env.xarm_api_left, "[left]")
        identify_apply_save(env.xarm_api_right, "[right]")
    except Exception as exc:  # noqa: BLE001
        print(f"Failed: {exc}", file=sys.stderr)
    finally:
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
