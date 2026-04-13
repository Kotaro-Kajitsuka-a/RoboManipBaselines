import argparse

import matplotlib.pyplot as plt
import numpy as np

from robo_manip_baselines.common import DataKey, RmbData

TITLE_FONTSIZE = 20
LABEL_FONTSIZE = 17
TICK_FONTSIZE = 14
LEGEND_FONTSIZE = 14


def parse_argument():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("rmb_file", type=str, help="path to .rmb or .hdf5 file")
    parser.add_argument(
        "--arm_idx",
        type=int,
        default=0,
        help="arm index to plot when wrench dim is 12 (0 or 1)",
    )
    parser.add_argument(
        "--step_slice",
        type=str,
        default=None,
        help="slice of env steps to plot, e.g. '100:200'",
    )
    return parser.parse_args()


class VisualizeSubstepWrench:
    def __init__(self, rmb_file, arm_idx, step_slice):
        self.rmb_file = rmb_file
        self.arm_idx = arm_idx
        self.step_slice = self._parse_step_slice(step_slice)

    @staticmethod
    def _parse_step_slice(step_slice):
        if step_slice is None:
            return slice(None)

        tokens = step_slice.split(":")
        assert (
            len(tokens) == 2
        ), "[VisualizeSubstepWrench] step_slice must be 'start:end'."
        start = None if tokens[0] == "" else int(tokens[0])
        stop = None if tokens[1] == "" else int(tokens[1])
        return slice(start, stop)

    def _load_wrench_key(self, rmb_data, key):
        if key not in rmb_data:
            return None

        wrench_seq = np.asarray(rmb_data[key][self.step_slice], dtype=np.float64)
        assert (
            wrench_seq.ndim == 3
        ), f"[{self.__class__.__name__}] Expected 3D wrench sequence for {key}, got {wrench_seq.shape}."

        wrench_dim = wrench_seq.shape[2]
        if wrench_dim == 6:
            return wrench_seq

        assert (
            wrench_dim == 12
        ), f"[{self.__class__.__name__}] Unsupported wrench dim for {key}: {wrench_dim}."
        assert self.arm_idx in (
            0,
            1,
        ), f"[{self.__class__.__name__}] arm_idx must be 0 or 1 for 12D wrench."
        start = 6 * self.arm_idx
        return wrench_seq[:, :, start : start + 6]

    def _load_substep_wrench(self):
        with RmbData(self.rmb_file, "r") as rmb_data:
            assert (
                DataKey.SUBSTEP_MEASURED_EEF_WRENCH in rmb_data
            ), f"[{self.__class__.__name__}] '{DataKey.SUBSTEP_MEASURED_EEF_WRENCH}' is not found: {self.rmb_file}"

            raw_wrench = self._load_wrench_key(
                rmb_data, DataKey.SUBSTEP_MEASURED_EEF_WRENCH
            )
            comp_wrench = self._load_wrench_key(
                rmb_data, DataKey.SUBSTEP_COMPENSATED_EEF_WRENCH
            )
            comp_lpf_wrench = self._load_wrench_key(
                rmb_data, DataKey.SUBSTEP_COMPENSATED_LPF_EEF_WRENCH
            )
            if DataKey.TIME in rmb_data:
                step_time = np.asarray(rmb_data[DataKey.TIME][self.step_slice])
            else:
                step_time = None

        return raw_wrench, comp_wrench, comp_lpf_wrench, step_time

    @staticmethod
    def _make_substep_time(step_time, num_substeps):
        if step_time is None:
            return np.arange(num_substeps, dtype=np.float64)

        step_time = np.asarray(step_time, dtype=np.float64)
        if len(step_time) >= 2:
            control_dt = float(np.median(np.diff(step_time)))
        else:
            control_dt = 1.0
        substep_dt = control_dt / num_substeps
        return (
            step_time[:, None]
            + substep_dt * np.arange(num_substeps, dtype=np.float64)[None, :]
        ).reshape(-1)

    def run(self):
        raw_wrench, comp_wrench, comp_lpf_wrench, step_time = (
            self._load_substep_wrench()
        )
        num_steps, num_substeps, _ = raw_wrench.shape
        raw_wrench = raw_wrench.reshape(num_steps * num_substeps, 6)
        if comp_wrench is not None:
            comp_wrench = comp_wrench.reshape(num_steps * num_substeps, 6)
        if comp_lpf_wrench is not None:
            comp_lpf_wrench = comp_lpf_wrench.reshape(num_steps * num_substeps, 6)
        time = self._make_substep_time(step_time, num_substeps)

        labels_force = ["Fx", "Fy", "Fz"]
        labels_torque = ["Nx", "Ny", "Nz"]

        fig, axes = plt.subplots(
            2, 1, figsize=(12, 7), sharex=True, constrained_layout=True
        )
        fig.suptitle(
            f"substep wrench ({self.rmb_file}, arm_idx={self.arm_idx})",
            fontsize=TITLE_FONTSIZE,
        )

        for i in range(3):
            axes[0].plot(time, raw_wrench[:, i], label=f"{labels_force[i]} raw")
            if comp_wrench is not None:
                axes[0].plot(
                    time, comp_wrench[:, i], "--", label=f"{labels_force[i]} comp"
                )
            if comp_lpf_wrench is not None:
                axes[0].plot(
                    time,
                    comp_lpf_wrench[:, i],
                    "-.",
                    label=f"{labels_force[i]} comp_lpf",
                )
        axes[0].set_ylabel("Force [N]", fontsize=LABEL_FONTSIZE)
        axes[0].grid(True)
        axes[0].legend(loc="upper right", ncol=3, fontsize=LEGEND_FONTSIZE)
        axes[0].tick_params(axis="both", labelsize=TICK_FONTSIZE)

        for i in range(3):
            axes[1].plot(time, raw_wrench[:, 3 + i], label=f"{labels_torque[i]} raw")
            if comp_wrench is not None:
                axes[1].plot(
                    time,
                    comp_wrench[:, 3 + i],
                    "--",
                    label=f"{labels_torque[i]} comp",
                )
            if comp_lpf_wrench is not None:
                axes[1].plot(
                    time,
                    comp_lpf_wrench[:, 3 + i],
                    "-.",
                    label=f"{labels_torque[i]} comp_lpf",
                )
        axes[1].set_xlabel(
            "Time [s]" if step_time is not None else "Substep",
            fontsize=LABEL_FONTSIZE,
        )
        axes[1].set_ylabel("Torque [Nm]", fontsize=LABEL_FONTSIZE)
        axes[1].grid(True)
        axes[1].legend(loc="upper right", ncol=3, fontsize=LEGEND_FONTSIZE)
        axes[1].tick_params(axis="both", labelsize=TICK_FONTSIZE)

        plt.show()


if __name__ == "__main__":
    visualize = VisualizeSubstepWrench(**vars(parse_argument()))
    visualize.run()
