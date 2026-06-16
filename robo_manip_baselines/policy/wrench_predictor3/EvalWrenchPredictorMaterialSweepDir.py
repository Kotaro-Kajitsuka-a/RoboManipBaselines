import argparse
import contextlib
import csv
import os
import sys

import numpy as np

from robo_manip_baselines.common import find_rmb_files
from robo_manip_baselines.misc.SummarizeWrenchPredictorMaterialSweepCsv import (
    summarize_material_sweep_csv,
)
from robo_manip_baselines.policy.wrench_predictor3.EvalWrenchPredictor import (
    EvalWrenchPredictor,
)
from robo_manip_baselines.policy.wrench_predictor3.MaterialPropertyUtils import (
    extract_material_object_key,
)


TARGET_OBJECT_KEYS = [
    "WrenchPredObject0",
    "WrenchPredObject1",
    "WrenchPredObject2",
    "WrenchPredObject3",
]


def parse_argument():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("checkpoint", type=str, help="checkpoint file")
    parser.add_argument(
        "rmb_dir",
        type=str,
        help="path to normal_0123 directory containing RMB episode files",
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


class EvalWrenchPredictorMaterialSweepDir:
    def __init__(self, checkpoint, rmb_dir):
        assert "normal_0123" in rmb_dir, rmb_dir
        self.checkpoint = checkpoint
        self.rmb_dir = rmb_dir

    def setup_output_dir(self):
        checkpoint_dir = os.path.dirname(self.checkpoint)
        rmb_dir_name = os.path.basename(os.path.normpath(self.rmb_dir))
        output_name = f"{rmb_dir_name}_material_sweep"
        self.output_dir = os.path.join(checkpoint_dir, "eval", output_name)
        os.makedirs(self.output_dir, exist_ok=True)

        checkpoint_stem = os.path.splitext(os.path.basename(self.checkpoint))[0]
        self.output_log = os.path.join(
            self.output_dir, f"{output_name}_{checkpoint_stem}_eval.log"
        )
        self.output_csv = os.path.join(
            self.output_dir, f"{output_name}_{checkpoint_stem}_eval.csv"
        )

    def run(self):
        self.setup_output_dir()
        with open(self.output_log, "w") as log_file:
            tee = Tee(sys.stdout, log_file)
            with contextlib.redirect_stdout(tee):
                self.run_eval()
                print(f"[{self.__class__.__name__}] Save log: {self.output_log}")
                print(f"[{self.__class__.__name__}] Save csv: {self.output_csv}")

    def run_eval(self):
        rmb_path_list = find_rmb_files(self.rmb_dir)
        assert len(rmb_path_list) > 0, self.rmb_dir

        evaluator = EvalWrenchPredictor(self.checkpoint, rmb_path_list[0])
        object_key_to_id = evaluator.model_meta_info["material_property"][
            "object_key_to_id"
        ]
        for object_key in TARGET_OBJECT_KEYS:
            assert object_key in object_key_to_id, object_key

        object_key_to_rmb_paths = self.group_rmb_paths_by_object_key(rmb_path_list)
        for object_key in TARGET_OBJECT_KEYS:
            assert object_key in object_key_to_rmb_paths, object_key

        print(f"[{self.__class__.__name__}] Output directory: {self.output_dir}")
        print(f"[{self.__class__.__name__}] Evaluate {len(rmb_path_list)} episodes.")
        print(
            f"[{self.__class__.__name__}] Actual objects: "
            f"{TARGET_OBJECT_KEYS}"
        )
        print(
            f"[{self.__class__.__name__}] Material properties: {TARGET_OBJECT_KEYS}"
        )

        rows = []
        for actual_object_key in TARGET_OBJECT_KEYS:
            actual_rmb_paths = object_key_to_rmb_paths[actual_object_key]
            print(f"[actual object: {actual_object_key}]")
            for material_object_key in TARGET_OBJECT_KEYS:
                row = self.evaluate_rmb_paths(
                    evaluator,
                    actual_object_key,
                    material_object_key,
                    actual_rmb_paths,
                )
                rows.append(row)
                self.print_row(row)

        self.save_csv(rows)
        summarize_material_sweep_csv(self.output_csv)

    def group_rmb_paths_by_object_key(self, rmb_path_list):
        object_key_to_rmb_paths = {}
        for rmb_path in rmb_path_list:
            object_key = extract_material_object_key(rmb_path)
            assert object_key is not None, rmb_path
            if object_key not in TARGET_OBJECT_KEYS:
                continue
            object_key_to_rmb_paths.setdefault(object_key, []).append(rmb_path)

        return object_key_to_rmb_paths

    def evaluate_rmb_paths(
        self,
        evaluator,
        actual_object_key,
        material_object_key,
        rmb_paths,
    ):
        abs_error_seq_list = []
        plot_dir = os.path.join(
            self.output_dir,
            f"actual_{actual_object_key}",
            f"material_{material_object_key}",
        )
        os.makedirs(plot_dir, exist_ok=True)

        for rmb_path in rmb_paths:
            evaluator.output_dir = plot_dir
            evaluator.set_rmb_filename(rmb_path)
            evaluator.material_object_key = material_object_key
            evaluator.setup_material_property()
            time_seq, gt_wrench_seq, pred_wrench_seq, abs_error_seq, mae = (
                evaluator.evaluate()
            )
            evaluator.save_plot(time_seq, gt_wrench_seq, pred_wrench_seq, mae)
            abs_error_seq_list.append(abs_error_seq)

        all_abs_error_seq = np.concatenate(abs_error_seq_list, axis=0)
        mae = np.mean(all_abs_error_seq, axis=0)
        return {
            "actual_object_key": actual_object_key,
            "material_object_key": material_object_key,
            "episode_count": len(rmb_paths),
            "Fx": mae[0],
            "Fy": mae[1],
            "Fz": mae[2],
            "Nx": mae[3],
            "Ny": mae[4],
            "Nz": mae[5],
            "Fx/Fy mean": mae[:2].mean(),
            "Fz only": mae[2],
            "torque mean": mae[3:].mean(),
            "plot_dir": plot_dir,
        }

    def print_row(self, row):
        is_correct = row["actual_object_key"] == row["material_object_key"]
        print(
            f"  - material={row['material_object_key']}, "
            f"correct={is_correct}, "
            f"episodes={row['episode_count']}, "
            f"Fx/Fy mean={row['Fx/Fy mean']:.6f}, "
            f"Fx={row['Fx']:.6f}, "
            f"Fy={row['Fy']:.6f}, "
            f"Fz={row['Fz']:.6f}, "
            f"torque mean={row['torque mean']:.6f}, "
            f"plot_dir={row['plot_dir']}"
        )

    def save_csv(self, rows):
        fieldnames = [
            "actual_object_key",
            "material_object_key",
            "episode_count",
            "Fx/Fy mean",
            "Fx",
            "Fy",
            "Fz",
            "Nx",
            "Ny",
            "Nz",
            "Fz only",
            "torque mean",
            "plot_dir",
        ]
        with open(self.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    evaluator = EvalWrenchPredictorMaterialSweepDir(**vars(parse_argument()))
    evaluator.run()
