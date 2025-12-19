#!/usr/bin/env python3
"""Print world pose of a frame/joint in the XArm7 URDF using Pinocchio.

Example:
  python robo_manip_baselines/misc/DebugTcpBallPose.py \
    --urdf robo_manip_baselines/envs/assets/common/robots/xarm7/xarm7_1305_left_ball_ee_right.urdf \
    --frame R_joint_tcp_ball \
    --q 0 0 0 0 0 0 0
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pinocchio as pin


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--urdf",
        default=(
            "robo_manip_baselines/envs/assets/common/robots/xarm7/"
            "xarm7_1305_left_ball_ee_right.urdf"
        ),
        help="Path to the URDF to load.",
    )
    parser.add_argument(
        "--frame",
        default="R_joint_tcp_ball",
        help="Frame name to query (e.g., R_joint_tcp_ball or R_link_tcp_ball).",
    )
    parser.add_argument(
        "--q",
        nargs="*",
        type=float,
        default=None,
        help="Joint configuration (nq values). If omitted, uses neutral.",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Use a random configuration instead of --q/neutral.",
    )
    parser.add_argument(
        "--list-frames",
        action="store_true",
        help="List frames that contain tcp_ball/tcp keywords.",
    )
    return parser.parse_args()


def _format_se3(se3: pin.SE3) -> str:
    quat = pin.Quaternion(se3.rotation).coeffs()[[3, 0, 1, 2]]  # qw, qx, qy, qz
    pos = se3.translation
    return f"pos={pos.tolist()}, quat(wxyz)={quat.tolist()}"


def main() -> int:
    args = _parse_args()
    urdf_path = Path(args.urdf)
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")

    model = pin.buildModelFromUrdf(str(urdf_path))
    data = model.createData()

    if args.list_frames:
        for i, frame in enumerate(model.frames):
            name = frame.name
            if "tcp" in name or "ball" in name:
                print(f"frame[{i}]: {name}, type={frame.type}, parent={frame.parent}")

    if args.random:
        q = pin.randomConfiguration(model)
    elif args.q is None:
        q = pin.neutral(model)
    else:
        q = np.array(args.q, dtype=np.float64)
        if q.size != model.nq:
            raise ValueError(f"--q length {q.size} != model.nq {model.nq}")

    # IMPORTANT: compute both joint and frame placements.
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)

    if not model.existFrame(args.frame):
        raise ValueError(f"Frame not found: {args.frame}")

    frame_id = model.getFrameId(args.frame)
    frame = model.frames[frame_id]
    oMf = data.oMf[frame_id]

    print(f"model.nq={model.nq}, model.nv={model.nv}")
    if hasattr(frame, "parentJoint"):
        parent_joint = frame.parentJoint
    else:
        parent_joint = frame.parent
    print(
        f"frame={frame.name}, id={frame_id}, type={frame.type}, parent_joint={parent_joint}"
    )
    print(f"frame.placement (parent->frame) = {_format_se3(frame.placement)}")
    print(f"oMf (world->frame) = {_format_se3(oMf)}")

    # Cross-check using parent joint placement: oMi * placement.
    oMi = data.oMi[parent_joint]
    oMf_check = oMi * frame.placement
    print(f"oMi (world->parent joint) = {_format_se3(oMi)}")
    print(f"oMi * placement = {_format_se3(oMf_check)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
