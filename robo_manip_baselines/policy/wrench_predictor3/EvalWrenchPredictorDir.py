import argparse
import contextlib
import os
import sys

import numpy as np

from robo_manip_baselines.common import find_rmb_files
from robo_manip_baselines.policy.wrench_predictor3.EvalWrenchPredictor import (
    EvalWrenchPredictor,
)


def parse_argument():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("checkpoint", type=str, help="checkpoint file")
    parser.add_argument(
        "rmb_dir",
        type=str,
        help="path to a directory containing RMB episode files",
    )
    parser.add_argument(
        "--material_property",
        type=str,
        default=None,
        help='fixed material property vector, e.g. "0.1 -0.2 ..."',
    )
    parser.add_argument(
        "--material_object_key",
        type=str,
        default=None,
        help="object key used to select material property from checkpoint (extracted from rmb_path by default)",
    )
    return parser.parse_args()


class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for file in self.files:
            file.write(data)
            file.flush()

    def flush(self):
        for file in self.files:
            file.flush()


class EvalWrenchPredictorDir:
    def __init__(
        self,
        checkpoint,
        rmb_dir,
        material_property=None,
        material_object_key=None,
    ):
        self.checkpoint = checkpoint
        self.rmb_dir = rmb_dir
        self.material_property = material_property
        self.material_object_key = material_object_key

    def setup_output_dir(self):
        checkpoint_dir = os.path.dirname(self.checkpoint)
        rmb_dir_parts = os.path.normpath(self.rmb_dir).split(os.sep)
        if "validation" in rmb_dir_parts:
            validation_idx = rmb_dir_parts.index("validation")
            assert validation_idx + 1 < len(rmb_dir_parts), self.rmb_dir
            output_name = rmb_dir_parts[validation_idx + 1]
        else:
            output_name = rmb_dir_parts[-1]
        self.output_dir = os.path.join(checkpoint_dir, "eval", output_name)
        os.makedirs(self.output_dir, exist_ok=True)

        checkpoint_stem = os.path.splitext(os.path.basename(self.checkpoint))[0]
        self.output_log = os.path.join(
            self.output_dir, f"{output_name}_{checkpoint_stem}_eval.log"
        )

    def run(self):
        self.setup_output_dir()
        with open(self.output_log, "w") as log_file:
            tee = Tee(sys.stdout, log_file)
            with contextlib.redirect_stdout(tee):
                self.run_eval()
                print(f"[{self.__class__.__name__}] Save log: {self.output_log}")

    def run_eval(self):
        rmb_path_list = find_rmb_files(self.rmb_dir)
        if len(rmb_path_list) == 0:
            raise ValueError(
                f"[{self.__class__.__name__}] RMB files are not found: {self.rmb_dir}"
            )

        evaluator = EvalWrenchPredictor(
            self.checkpoint,
            rmb_path_list[0],
            self.material_property,
            self.material_object_key,
        )
        evaluator.output_dir = self.output_dir
        labels = ["Fx", "Fy", "Fz", "Nx", "Ny", "Nz"]
        abs_error_seq_list = []

        print(f"[{self.__class__.__name__}] Output directory: {self.output_dir}")
        print(f"[{self.__class__.__name__}] Evaluate {len(rmb_path_list)} episodes.")
        for rmb_path in rmb_path_list:
            evaluator.set_rmb_filename(rmb_path)
            evaluator.setup_material_property()
            time_seq, gt_wrench_seq, pred_wrench_seq, abs_error_seq, mae = (
                evaluator.evaluate()
            )
            evaluator.save_plot(time_seq, gt_wrench_seq, pred_wrench_seq, mae)
            abs_error_seq_list.append(abs_error_seq)

            rmb_name = os.path.basename(rmb_path.rstrip("/"))
            print(f"[{self.__class__.__name__}] {rmb_name} MAE:")
            for label, value in zip(labels, mae):
                print(f"  - {label}: {value:.6f}")
            print(f"  - mean: {mae.mean():.6f}")

        all_abs_error_seq = np.concatenate(abs_error_seq_list, axis=0)
        all_mae = np.mean(all_abs_error_seq, axis=0)
        print(f"[{self.__class__.__name__}] All episodes MAE:")
        for label, value in zip(labels, all_mae):
            print(f"  - {label}: {value:.6f}")
        print(f"  - mean: {all_mae.mean():.6f}")


if __name__ == "__main__":
    evaluator = EvalWrenchPredictorDir(**vars(parse_argument()))
    evaluator.run()
