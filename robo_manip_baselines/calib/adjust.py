import argparse
from pathlib import Path

import numpy as np


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
    回転のみを調整する。

    現在は:
      1. Local X : -1 deg
      2. Local Z : -1 deg
    """

    # R_current = T[:3, :3]

    # rot_local = (
    #     #R.from_euler("x", -1.0, degrees=True)
    #     #* R.from_euler("z", -1.0, degrees=True)
    # ).as_matrix()

    # # Local座標系で回転を適用
    # R_new = R_current @ rot_local

    # T_new = T.copy()
    # T_new[:3, :3] = R_new

    # return T_new
    return T


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
