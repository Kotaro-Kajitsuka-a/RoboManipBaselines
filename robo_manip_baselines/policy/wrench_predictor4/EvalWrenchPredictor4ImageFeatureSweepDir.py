import csv
import os

import numpy as np
import torch
from tqdm import tqdm

from robo_manip_baselines.common import denormalize_data
from robo_manip_baselines.policy.wrench_predictor4.EvalWrenchPredictor4SweepCommon import (
    WRENCH_LABELS,
    EvalWrenchPredictor4SweepBase,
    parse_sweep_argument,
    plt,
)


class EvalWrenchPredictor4ImageFeatureSweepDir(EvalWrenchPredictor4SweepBase):
    def get_output_name(self, rmb_dir_name):
        return f"{rmb_dir_name}_wrench_predictor4_image_feature_sweep"

    def get_output_csv_name(self):
        return "image_feature_sweep_eval.csv"

    def get_summary_csv_name(self):
        return "image_feature_sweep_matrix_summary.csv"

    def get_diagonal_accuracy_csv_name(self):
        return "image_feature_sweep_diagonal_accuracy.csv"

    def get_heatmap_png_name(self, checkpoint_stem):
        return f"{checkpoint_stem}_wrench_predictor4_image_feature_sweep_heatmap.png"

    def get_heatmap_metrics(self):
        return [
            ("normalized image feature MSE", "normalized image feature MSE"),
            ("normalized image feature MAE", "normalized image feature mean"),
            ("image feature MAE", "image feature mean"),
            ("image feature RMSE", "image feature RMSE"),
            ("auxiliary force MAE [N]", "force mean"),
            ("auxiliary torque MAE [N m]", "torque mean"),
            ("normalized wrench error", "normalized wrench mean"),
            ("normalized total error", "normalized total mean"),
        ]

    def evaluate_object_pair(
        self,
        checkpoint,
        actual_object_key,
        material_object_key,
        filenames,
    ):
        _dataset, dataloader = self.make_dataloader(filenames)
        material_object_id = self.object_key_to_id[material_object_key]
        wrench_abs_error_list = []
        image_feature_abs_error_list = []
        total_abs_error_list = []

        for batch in tqdm(
            dataloader,
            desc=(
                f"{os.path.basename(checkpoint)} "
                f"{actual_object_key} <- {material_object_key}"
            ),
            leave=False,
        ):
            batch = self.move_batch_to_device(batch, material_object_id)
            with torch.inference_mode():
                pred = self.policy.predict(batch)
            wrench_abs_error, image_feature_abs_error, total_abs_error = (
                self.compute_abs_error(batch, pred)
            )
            wrench_abs_error_list.append(wrench_abs_error)
            image_feature_abs_error_list.append(image_feature_abs_error)
            total_abs_error_list.append(total_abs_error)

        wrench_abs_error = np.concatenate(wrench_abs_error_list)
        image_feature_abs_error = np.concatenate(image_feature_abs_error_list)
        total_abs_error = np.concatenate(total_abs_error_list)
        wrench_mae = wrench_abs_error.mean(axis=0)
        image_feature_rmse = np.sqrt(
            np.mean(np.square(image_feature_abs_error), axis=-1)
        )
        wrench_dim = self.model_meta_info["policy"]["args"]["wrench_dim"]
        normalized_wrench_abs_error = total_abs_error[:, :wrench_dim]
        normalized_image_feature_abs_error = total_abs_error[:, wrench_dim:]

        return {
            "checkpoint": os.path.basename(checkpoint),
            "actual_object_key": actual_object_key,
            "material_object_key": material_object_key,
            "is_correct_material": actual_object_key == material_object_key,
            "episode_count": len(filenames),
            "sample_count": len(wrench_abs_error),
            "image feature mean": image_feature_abs_error.mean(),
            "image feature RMSE": image_feature_rmse.mean(),
            "Fx": wrench_mae[0],
            "Fy": wrench_mae[1],
            "Fz": wrench_mae[2],
            "Nx": wrench_mae[3],
            "Ny": wrench_mae[4],
            "Nz": wrench_mae[5],
            "force mean": wrench_mae[:3].mean(),
            "torque mean": wrench_mae[3:].mean(),
            "wrench mean": wrench_mae.mean(),
            "normalized image feature mean": normalized_image_feature_abs_error.mean(),
            "normalized image feature MSE": np.square(
                normalized_image_feature_abs_error
            ).mean(),
            "normalized wrench mean": normalized_wrench_abs_error.mean(),
            "normalized total mean": total_abs_error.mean(),
        }

    def print_row(self, row):
        print(
            f"  - ckpt={row['checkpoint']}, "
            f"actual={row['actual_object_key']}, "
            f"PB={row['material_object_key']}, "
            f"correct={row['is_correct_material']}, "
            f"feature={row['image feature mean']:.6f}, "
            f"force={row['force mean']:.6f} N, "
            f"torque={row['torque mean']:.6f} N m"
        )

    def save_csv(self, rows):
        fieldnames = [
            "checkpoint",
            "actual_object_key",
            "material_object_key",
            "is_correct_material",
            "episode_count",
            "sample_count",
            "image feature mean",
            "image feature RMSE",
            "Fx",
            "Fy",
            "Fz",
            "Nx",
            "Ny",
            "Nz",
            "force mean",
            "torque mean",
            "wrench mean",
            "normalized image feature mean",
            "normalized image feature MSE",
            "normalized wrench mean",
            "normalized total mean",
        ]
        with open(self.output_csv, "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def save_episode_plots(self, checkpoint, actual_object_key, filenames):
        checkpoint_stem = os.path.splitext(os.path.basename(checkpoint))[0]
        plot_dir = os.path.join(
            self.episode_dir,
            checkpoint_stem,
            f"actual_{actual_object_key}",
        )
        os.makedirs(plot_dir, exist_ok=True)

        for filename in tqdm(
            filenames,
            desc=f"{checkpoint_stem} {actual_object_key} episode plots",
            leave=False,
        ):
            plot_data = self.evaluate_episode_for_plot(filename)
            rmb_stem = os.path.basename(filename.rstrip("/")).replace(".rmb", "")
            output_png = os.path.join(
                plot_dir,
                f"{rmb_stem}_hstep_wrench_image_feature_loss.png",
            )
            self.save_episode_plot(
                output_png,
                checkpoint_stem,
                actual_object_key,
                rmb_stem,
                plot_data,
            )

    def evaluate_episode_for_plot(self, filename):
        dataset, dataloader = self.make_dataloader([filename])
        final_time_idx = self.get_final_time_idx(filename, dataset)
        gt_wrench = None
        material_key_to_pred_wrench = {}
        material_key_to_normalized_feature_mse = {}
        material_key_to_normalized_feature_mae = {}

        for material_object_key in self.material_object_keys:
            material_object_id = self.object_key_to_id[material_object_key]
            gt_wrench_list = []
            pred_wrench_list = []
            normalized_feature_mse_list = []
            normalized_feature_mae_list = []
            for batch in dataloader:
                batch = self.move_batch_to_device(batch, material_object_id)
                with torch.inference_mode():
                    pred = self.policy.predict(batch)
                batch_result = self.compute_hstep_plot_data(batch, pred)
                gt_wrench_list.append(batch_result[0])
                pred_wrench_list.append(batch_result[1])
                normalized_feature_mse_list.append(batch_result[2])
                normalized_feature_mae_list.append(batch_result[3])

            if gt_wrench is None:
                gt_wrench = np.concatenate(gt_wrench_list)
            material_key_to_pred_wrench[material_object_key] = np.concatenate(
                pred_wrench_list
            )
            material_key_to_normalized_feature_mse[material_object_key] = (
                np.concatenate(normalized_feature_mse_list)
            )
            material_key_to_normalized_feature_mae[material_object_key] = (
                np.concatenate(normalized_feature_mae_list)
            )

        return {
            "time_idx": final_time_idx,
            "gt_wrench": gt_wrench,
            "material_key_to_pred_wrench": material_key_to_pred_wrench,
            "material_key_to_normalized_feature_mse": (
                material_key_to_normalized_feature_mse
            ),
            "material_key_to_normalized_feature_mae": (
                material_key_to_normalized_feature_mae
            ),
        }

    def compute_hstep_plot_data(self, batch, pred):
        gt_wrench = denormalize_data(
            batch["wrench"][:, -1].detach().cpu().numpy(),
            self.model_meta_info["wrench"],
        )
        pred_wrench = denormalize_data(
            pred["wrench"][:, -1].detach().cpu().numpy(),
            self.model_meta_info["wrench"],
        )
        normalized_image_feature_abs_error = np.abs(
            pred["image_feature"][:, -1].detach().cpu().numpy()
            - batch["image_feature"][:, -1].detach().cpu().numpy()
        )
        normalized_feature_mse = np.square(
            normalized_image_feature_abs_error
        ).mean(axis=-1)
        normalized_feature_mae = normalized_image_feature_abs_error.mean(axis=-1)
        return gt_wrench, pred_wrench, normalized_feature_mse, normalized_feature_mae

    def save_episode_plot(
        self,
        output_png,
        checkpoint_stem,
        actual_object_key,
        rmb_stem,
        plot_data,
    ):
        fig, axes = plt.subplots(8, 1, figsize=(12, 18), sharex=True)
        time_idx = plot_data["time_idx"]
        for wrench_idx, ax in enumerate(axes[:6]):
            ax.plot(
                time_idx,
                plot_data["gt_wrench"][:, wrench_idx],
                color="black",
                linewidth=2.0,
                label=f"GT {WRENCH_LABELS[wrench_idx]}",
            )
            for material_object_key, pred_wrench in plot_data[
                "material_key_to_pred_wrench"
            ].items():
                ax.plot(
                    time_idx,
                    pred_wrench[:, wrench_idx],
                    linewidth=1.2,
                    label=f"{material_object_key} PB",
                )
            ax.set_ylabel(WRENCH_LABELS[wrench_idx])
            ax.grid(True)
            ax.legend(loc="best", fontsize=8)

        for ax, data_key, ylabel in (
            (
                axes[6],
                "material_key_to_normalized_feature_mse",
                "normalized feature MSE",
            ),
            (
                axes[7],
                "material_key_to_normalized_feature_mae",
                "normalized feature MAE",
            ),
        ):
            for material_object_key, error in plot_data[data_key].items():
                ax.plot(
                    time_idx,
                    error,
                    linewidth=1.2,
                    label=f"{material_object_key} PB",
                )
            ax.set_ylabel(ylabel)
            ax.grid(True)
            ax.legend(loc="best", fontsize=8)

        axes[-1].set_xlabel("skipped time index of t + H - 1")
        fig.suptitle(
            f"{checkpoint_stem} / actual={actual_object_key} / {rmb_stem}",
            fontsize=11,
        )
        fig.tight_layout()
        fig.savefig(output_png)
        plt.close(fig)


if __name__ == "__main__":
    evaluator = EvalWrenchPredictor4ImageFeatureSweepDir(**vars(parse_sweep_argument()))
    evaluator.run()
