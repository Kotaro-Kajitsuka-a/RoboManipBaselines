import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R


def load_calib(path):
    data = np.loadtxt(path)

    if data.shape == (3, 4):
        T = np.eye(4)
        T[:3, :4] = data
    elif data.shape == (4, 4):
        T = data
    else:
        raise ValueError(f"Unexpected calib shape: {data.shape}")

    return T


def save_calib(path, T):
    np.savetxt(path, T, fmt="%.8f")


def adjust_transform(T):
    """
    calibに入っている R_base_camera に対して、
    cameraローカル座標系で回転補正をかける。
    """

    R_base_camera = T[:3, :3]

    # camera <- base
    R_camera_base = R_base_camera.T

    rot_local = (R.from_euler("x", -40.0, degrees=True)).as_matrix()

    # base 座標系で回転
    R_camera_base_new = R_camera_base @ rot_local

    # 元の形式(base <- camera)に戻す
    R_base_camera_new = R_camera_base_new.T

    T_new = T.copy()
    T_new[:3, :3] = R_base_camera_new

    return T_new


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("calib_file", help="Input .calib file")
    args = parser.parse_args()

    input_path = Path(args.calib_file)

    if input_path.suffix != ".calib":
        raise ValueError("Input must be a .calib file")

    output_path = input_path.with_name(input_path.stem + "_adjusted.calib")

    T = load_calib(input_path)
    T_new = adjust_transform(T)

    save_calib(output_path, T_new)

    print("Input:")
    print(T)

    print("\nAdjusted:")
    print(T_new)

    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
