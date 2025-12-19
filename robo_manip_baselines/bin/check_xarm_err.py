#!/usr/bin/env python3
"""Quick helper to inspect xArm error/warn codes without moving the robot.

Usage examples:
    python robo_manip_baselines/bin/check_xarm_err.py --ip-left 192.168.1.208
    python robo_manip_baselines/bin/check_xarm_err.py --ip-right 192.168.1.209
    python robo_manip_baselines/bin/check_xarm_err.py --ip-left 192.168.1.208 --ip-right 192.168.1.209

This only calls `get_err_warn_code()` and prints mode/state so you can see
which arm is reporting what (e.g., err=22 means self-collision per SDK docs).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple


def _inject_local_sdk() -> None:
    """Ensure the local xArm-Python-SDK checkout is importable."""
    repo_root = Path(__file__).resolve().parents[2]
    sdk_root = repo_root / "xArm-Python-SDK"
    if sdk_root.exists():
        sys.path.insert(0, str(sdk_root))


_inject_local_sdk()

try:
    from xarm.wrapper import XArmAPI  # type: ignore
except Exception as exc:  # pragma: no cover - runtime import helper
    sys.stderr.write(
        "Failed to import xArm SDK. Install it (pip install xarm-python-sdk) "
        "or keep the xArm-Python-SDK directory in the repo root.\n"
    )
    raise


def _format_err_warn(api: XArmAPI, label: str) -> str:
    code, err_warn = api.get_err_warn_code()
    err_code, warn_code = (
        err_warn if isinstance(err_warn, (list, tuple)) and len(err_warn) >= 2 else (None, None)
    )
    mode = getattr(api, "mode", None)
    state_attr = getattr(api, "state", None)
    state_code, state_val = api.get_state()
    return (
        f"[{label}] get_err_warn_code -> code={code}, err={err_code}, "
        f"warn={warn_code}, mode={mode}, state_attr={state_attr}, "
        f"get_state={state_code}/{state_val}"
    )


def _dump_detail(api: XArmAPI, label: str) -> None:
    detail_calls = [
        ("c23 joint angle limit", lambda: api.get_c23_error_info(is_radian=True)),
        ("c24 joint speed limit", lambda: api.get_c24_error_info(is_radian=True)),
        ("c38 hard angle limit", lambda: api.get_c38_error_info(is_radian=True)),
        ("c31 collision torque", api.get_c31_error_info),
        ("c37 payload error", lambda: api.get_c37_error_info(is_radian=True)),
        ("c60 linear speed limit", api.get_c60_error_info),
    ]

    for name, fn in detail_calls:
        try:
            code, info = fn()
            print(f"[{label}] {name}: code={code}, info={info}")
        except Exception as exc:  # pragma: no cover - informative only
            print(f"[{label}] {name}: failed ({exc})")


def _check_arm(ip: str, label: str) -> Tuple[int, Optional[int], Optional[int]]:
    api = XArmAPI(ip)
    api.connect()
    print(f"Connected to {label} ({ip}).")
    print(_format_err_warn(api, label))
    _dump_detail(api, label)
    return api.disconnect()


def main() -> None:
    parser = argparse.ArgumentParser(description="Print xArm get_err_warn_code for given IPs")
    parser.add_argument("--ip-left", help="Left arm IP address")
    parser.add_argument("--ip-right", help="Right arm IP address")
    args = parser.parse_args()

    if not args.ip_left and not args.ip_right:
        parser.error("At least one of --ip-left or --ip-right is required")

    if args.ip_left:
        _check_arm(args.ip_left, "left")
    if args.ip_right:
        _check_arm(args.ip_right, "right")


if __name__ == "__main__":
    main()
