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
    viewerで描画している base center frame の原点位置は固定し、
    base center frame のローカルx軸まわりに回転補正をかける。
    """

    # calib は base <- camera として使われる。
    # viewer の drawFrameAxes はその逆の camera <- base を描くので、
    # まず camera <- base に直してから、原点 tvec を固定したまま回す。
    T_camera_base = np.linalg.inv(T)

    R_camera_base = T_camera_base[:3, :3]
    t_camera_base = T_camera_base[:3, 3].copy()

    rot_local = (R.from_euler("x", -0.90, degrees=True)).as_matrix()

    # base center frame のローカルx軸まわりに回す。
    R_camera_base_new = R_camera_base @ rot_local

    T_camera_base_new = np.eye(4)
    T_camera_base_new[:3, :3] = R_camera_base_new
    T_camera_base_new[:3, 3] = t_camera_base

    # 元の calib 形式(base <- camera)に戻す。
    return np.linalg.inv(T_camera_base_new)


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
