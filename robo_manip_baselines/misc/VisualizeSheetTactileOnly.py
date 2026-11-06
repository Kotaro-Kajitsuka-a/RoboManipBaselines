import argparse

import matplotlib.pyplot as plt
import numpy as np

from robo_manip_baselines.common import DataManager
from robo_manip_baselines.common.data.DataKey import DataKey


def parse_argument():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("rmb_filename", type=str)
    parser.add_argument("--sheet_key", type=str, default="sheet")
    parser.add_argument("--skip", type=int, default=1)
    parser.add_argument("--interval", type=float, default=0.1)
    return parser.parse_args()


def main():
    args = parse_argument()
    if args.skip < 1:
        raise ValueError("--skip must be at least 1.")
    if args.interval < 0:
        raise ValueError("--interval must be non-negative.")

    data_manager = DataManager(env=None)
    data_manager.load_data(args.rmb_filename, load_keys=[args.sheet_key, DataKey.TIME])
    if args.sheet_key not in data_manager.all_data_seq:
        raise KeyError(f"Sheet key '{args.sheet_key}' was not found in the RMB data.")

    matrices = np.asarray(data_manager.get_data_seq(args.sheet_key))
    if matrices.ndim != 3 or len(matrices) == 0:
        raise ValueError(
            f"Expected '{args.sheet_key}' to contain a non-empty sequence of 2D matrices."
        )
    times = data_manager.all_data_seq.get(DataKey.TIME)
    if times is not None and len(times) != len(matrices):
        times = None

    plt.ion()
    fig, ax = plt.subplots()
    image = None
    quit_requested = False

    def handle_key(event):
        nonlocal quit_requested
        if event.key in ("q", "escape"):
            quit_requested = True

    fig.canvas.mpl_connect("key_press_event", handle_key)
    try:
        for frame_idx in range(0, len(matrices), args.skip):
            if quit_requested or not plt.fignum_exists(fig.number):
                break

            if image is None:
                image = ax.imshow(
                    matrices[frame_idx][::-1, :],
                    cmap="turbo",
                    vmin=0,
                    vmax=255,
                    interpolation="nearest",
                )
                plt.colorbar(image, ax=ax)
            else:
                image.set_data(matrices[frame_idx][::-1, :])

            title = f"Frame {frame_idx + 1}/{len(matrices)}"
            if times is not None:
                title += f" ({times[frame_idx]:.2f} s)"
            ax.set_title(title)
            plt.pause(args.interval)
    finally:
        plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
