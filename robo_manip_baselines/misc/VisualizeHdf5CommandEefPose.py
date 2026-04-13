import argparse

import h5py
import matplotlib.pyplot as plt
import numpy as np


def parse_argument():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("hdf5_file", type=str, help="path to .hdf5 file")
    parser.add_argument(
        "--arm_idx",
        type=int,
        default=0,
        help="arm index to plot when command_eef_pose dim is 14 (0 or 1)",
    )
    parser.add_argument(
        "--marker_size",
        type=float,
        default=8.0,
        help="marker size for scatter plot",
    )
    return parser.parse_args()


class VisualizeHdf5CommandEefPose:
    def __init__(self, hdf5_file, arm_idx, marker_size):
        self.hdf5_file = hdf5_file
        self.arm_idx = arm_idx
        self.marker_size = marker_size

    def _select_arm_pose(self, pose, key_name):
        if pose.ndim != 2:
            raise ValueError(
                f"[{self.__class__.__name__}] {key_name} must be 2D, got {pose.shape}"
            )

        if pose.shape[1] == 7:
            return pose
        if pose.shape[1] == 14:
            if self.arm_idx not in (0, 1):
                raise ValueError(
                    f"[{self.__class__.__name__}] arm_idx must be 0 or 1 for 14D {key_name}, got {self.arm_idx}"
                )
            start = 7 * self.arm_idx
            return pose[:, start : start + 7]

        raise ValueError(
            f"[{self.__class__.__name__}] Unsupported {key_name} dim: {pose.shape[1]} (expected 7 or 14)"
        )

    def _load_pose(self):
        with h5py.File(self.hdf5_file, "r") as h5file:
            if "command_eef_pose" not in h5file:
                raise KeyError(
                    f"[{self.__class__.__name__}] 'command_eef_pose' is not found: {self.hdf5_file}"
                )
            if "measured_eef_pose" not in h5file:
                raise KeyError(
                    f"[{self.__class__.__name__}] 'measured_eef_pose' is not found: {self.hdf5_file}"
                )

            command_eef_pose = np.asarray(
                h5file["command_eef_pose"][:], dtype=np.float64
            )
            measured_eef_pose = np.asarray(
                h5file["measured_eef_pose"][:], dtype=np.float64
            )

            if "time" in h5file:
                time = np.asarray(h5file["time"][:], dtype=np.float64)
            else:
                time = np.arange(command_eef_pose.shape[0], dtype=np.float64)

        command_eef_pose = self._select_arm_pose(command_eef_pose, "command_eef_pose")
        measured_eef_pose = self._select_arm_pose(
            measured_eef_pose, "measured_eef_pose"
        )
        return time, command_eef_pose, measured_eef_pose

    def run(self):
        time, command_eef_pose, measured_eef_pose = self._load_pose()

        fig, axes = plt.subplots(
            2, 1, figsize=(12, 7), sharex=True, constrained_layout=True
        )
        fig.suptitle(f"command/measured eef_pose ({self.hdf5_file})")

        labels_pos = ["tx", "ty", "tz"]
        labels_quat = ["qw", "qx", "qy", "qz"]

        for i in range(3):
            axes[0].scatter(
                time,
                command_eef_pose[:, i],
                s=self.marker_size,
                label=f"{labels_pos[i]} command",
            )
            axes[0].scatter(
                time,
                measured_eef_pose[:, i],
                s=self.marker_size,
                marker="x",
                label=f"{labels_pos[i]} measured",
            )
        axes[0].set_ylabel("Position [m]")
        axes[0].grid(True)
        axes[0].legend(loc="upper right")

        for i in range(4):
            axes[1].scatter(
                time,
                command_eef_pose[:, 3 + i],
                s=self.marker_size,
                label=f"{labels_quat[i]} command",
            )
            axes[1].scatter(
                time,
                measured_eef_pose[:, 3 + i],
                s=self.marker_size,
                marker="x",
                label=f"{labels_quat[i]} measured",
            )
        axes[1].set_xlabel("Time [s]" if np.any(np.diff(time) != 1.0) else "Step")
        axes[1].set_ylabel("Quaternion [-]")
        axes[1].grid(True)
        axes[1].legend(loc="upper right")

        plt.show()


if __name__ == "__main__":
    visualize = VisualizeHdf5CommandEefPose(**vars(parse_argument()))
    visualize.run()
