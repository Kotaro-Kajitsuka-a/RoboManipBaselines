import argparse
import os

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
        help="fixed material property vector, e.g. \"0.1 -0.2 ...\"",
    )
    parser.add_argument(
        "--material_object_key",
        type=str,
        default=None,
        help="object key used to select material property from checkpoint (extracted from rmb_path by default)",
    )
    return parser.parse_args()


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

    def run(self):
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
        labels = ["Fx", "Fy", "Fz", "Nx", "Ny", "Nz"]
        abs_error_seq_list = []

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
