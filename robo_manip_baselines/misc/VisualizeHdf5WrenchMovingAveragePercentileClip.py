import argparse

import h5py
import matplotlib.pyplot as plt
import numpy as np

from robo_manip_baselines.common import DataKey

TITLE_FONTSIZE = 20
LABEL_FONTSIZE = 17
TICK_FONTSIZE = 14
LEGEND_FONTSIZE = 14


def parse_argument():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("hdf5_file", type=str, help="path to .hdf5 file")
    parser.add_argument(
        "--arm_idx",
        type=int,
        default=0,
        help="arm index to plot when wrench dim is 12 (0 or 1)",
    )
    return parser.parse_args()


class VisualizeHdf5WrenchMovingAveragePercentileClip:
    def __init__(self, hdf5_file, arm_idx):
        self.hdf5_file = hdf5_file
        self.arm_idx = arm_idx
        self.wrench_key = DataKey.MEASURED_EEF_WRENCH_MOVING_AVERAGE_PERCENTILE_CLIP

    def _load_wrench(self):
        with h5py.File(self.hdf5_file, "r") as h5file:
            if self.wrench_key not in h5file:
                raise KeyError(
                    f"[{self.__class__.__name__}] '{self.wrench_key}' is not found: {self.hdf5_file}"
                )

            wrench = np.asarray(h5file[self.wrench_key][:], dtype=np.float64)

            if DataKey.TIME in h5file:
                time = np.asarray(h5file[DataKey.TIME][:], dtype=np.float64)
            else:
                time = np.arange(wrench.shape[0], dtype=np.float64)

        if wrench.ndim != 2:
            raise ValueError(
                f"[{self.__class__.__name__}] {self.wrench_key} must be 2D, got {wrench.shape}"
            )

        if wrench.shape[1] == 6:
            return time, wrench
        if wrench.shape[1] == 12:
            if self.arm_idx not in (0, 1):
                raise ValueError(
                    f"[{self.__class__.__name__}] arm_idx must be 0 or 1 for 12D wrench, got {self.arm_idx}"
                )
            start = 6 * self.arm_idx
            return time, wrench[:, start : start + 6]

        raise ValueError(
            f"[{self.__class__.__name__}] Unsupported wrench dim: {wrench.shape[1]} (expected 6 or 12)"
        )

    def run(self):
        time, wrench = self._load_wrench()
        labels_force = ["Fx", "Fy", "Fz"]
        labels_torque = ["Nx", "Ny", "Nz"]

        fig, axes = plt.subplots(
            2, 1, figsize=(12, 7), sharex=True, constrained_layout=True
        )
        fig.suptitle(
            f"{self.wrench_key} ({self.hdf5_file})",
            fontsize=TITLE_FONTSIZE,
        )

        for i in range(3):
            axes[0].plot(time, wrench[:, i], label=labels_force[i])
        axes[0].set_ylabel("Force [N]", fontsize=LABEL_FONTSIZE)
        axes[0].grid(True)
        axes[0].legend(loc="upper right", fontsize=LEGEND_FONTSIZE)
        axes[0].tick_params(axis="both", labelsize=TICK_FONTSIZE)

        for i in range(3):
            axes[1].plot(time, wrench[:, 3 + i], label=labels_torque[i])
        axes[1].set_xlabel(
            "Time [s]" if np.any(np.diff(time) != 1.0) else "Step",
            fontsize=LABEL_FONTSIZE,
        )
        axes[1].set_ylabel("Torque [Nm]", fontsize=LABEL_FONTSIZE)
        axes[1].grid(True)
        axes[1].legend(loc="upper right", fontsize=LEGEND_FONTSIZE)
        axes[1].tick_params(axis="both", labelsize=TICK_FONTSIZE)

        plt.show()


if __name__ == "__main__":
    visualize = VisualizeHdf5WrenchMovingAveragePercentileClip(**vars(parse_argument()))
    visualize.run()
