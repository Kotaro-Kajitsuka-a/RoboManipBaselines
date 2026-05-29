import argparse
from pathlib import Path

import numpy as np


LEFT_TO_CENTER_OFFSET_M = np.array([0.0, -0.3291, 0.0])
DEFAULT_OUTPUT_PATH = Path("robo_manip_baselines/calib/base_center_T.calib")


def load_calib(path):
    """
    calib ファイルを読み込み、4x4 の同次変換行列を返す。
    3x4 しか書かれていない形式に対応（最後の行は [0 0 0 1] を補う）。
    """
    data = np.loadtxt(path)
    if data.shape == (3, 4):
        T = np.eye(4)
        T[:3, :4] = data
    elif data.shape == (4, 4):
        T = data
    else:
        raise ValueError(f"Unexpected shape in {path}: {data.shape}")
    return T


def compute_center_transform_from_left(left_path):
    T_left = load_calib(left_path)
    T_center = T_left.copy()

    # calib は camera 座標を base 座標へ写す行列として使われている。
    # center 原点を left 原点から left 座標系の y=-0.3291 m に置くので、
    # camera -> center の並進成分は p_center = p_left - offset になる。
    T_center[:3, 3] = T_left[:3, 3] - LEFT_TO_CENTER_OFFSET_M
    return T_center


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create base_center_T.calib from a user-provided left-base calib file."
    )
    parser.add_argument(
        "left_calib_path",
        type=Path,
        help="Path to the left-base calib file to use as the only input.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output path. Default: {DEFAULT_OUTPUT_PATH}",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    T_center = compute_center_transform_from_left(args.left_calib_path)
    print("T_camera_to_center:")
    print(T_center)

    np.savetxt(args.output, T_center, fmt="%.8f")
    print(f"\nSaved center transform to: {args.output}")
