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
from robo_manip_baselines.policy.wrench_predictor5.WrenchPredictor5Model import (
    WrenchPredictor5Model,
)


class EvalWrenchPredictor5SweepDir(EvalWrenchPredictor4SweepBase):
    def setup_model_meta_info(self):
        super().setup_model_meta_info()
        assert self.model_meta_info["policy"]["name"] == "WrenchPredictor5"

    def get_output_name(self, rmb_dir_name):
        return f"{rmb_dir_name}_wrench_predictor5_sweep"

    def get_output_csv_name(self):
        return "sweep_eval.csv"

    def get_summary_csv_name(self):
        return "sweep_matrix_summary.csv"

    def get_diagonal_accuracy_csv_name(self):
        return "sweep_diagonal_accuracy.csv"

    def get_heatmap_png_name(self, checkpoint_stem):
        return f"{checkpoint_stem}_wrench_predictor5_sweep_heatmap.png"

    def get_heatmap_metrics(self):
        return [
            ("SD3 latent MAE", "SD3 latent mean"),
            ("auxiliary force MAE [N]", "force mean"),
            ("auxiliary torque MAE [N m]", "torque mean"),
            ("normalized SD3 latent error", "normalized image feature mean"),
            ("normalized wrench error", "normalized wrench mean"),
            ("normalized total error", "normalized total mean"),
        ]

    def setup_policy(self, checkpoint):
        self.policy = WrenchPredictor5Model(**self.model_meta_info["policy"]["args"])
        self.policy.load_state_dict(
            torch.load(checkpoint, map_location=self.device, weights_only=True)
        )
        self.policy.to(self.device)
        self.policy.eval()

    def compute_abs_error(self, batch, pred):
        start = self.model_meta_info["data"]["n_obs_steps"]
        gt_wrench_normalized = batch["wrench"][:, start:].detach().cpu().numpy()
        pred_wrench_normalized = pred["wrench"].detach().cpu().numpy()
        gt_image_feature_normalized = (
            batch["image_feature"][:, start:].detach().cpu().numpy()
        )
        pred_image_feature_normalized = pred["image_feature"].detach().cpu().numpy()
        assert pred_wrench_normalized.shape == gt_wrench_normalized.shape
        assert pred_image_feature_normalized.shape == gt_image_feature_normalized.shape

        normalized_wrench_abs_error = np.abs(
            pred_wrench_normalized - gt_wrench_normalized
        ).reshape(-1, self.model_meta_info["policy"]["args"]["wrench_dim"])
        normalized_image_feature_abs_error = np.abs(
            pred_image_feature_normalized - gt_image_feature_normalized
        ).reshape(
            -1,
            self.model_meta_info["policy"]["args"]["image_feature_dim"],
        )

        gt_wrench = denormalize_data(
            gt_wrench_normalized,
            self.model_meta_info["wrench"],
        )
        pred_wrench = denormalize_data(
            pred_wrench_normalized,
            self.model_meta_info["wrench"],
        )
        gt_image_feature = denormalize_data(
            gt_image_feature_normalized,
            self.model_meta_info["image_feature"],
        )
        pred_image_feature = denormalize_data(
            pred_image_feature_normalized,
            self.model_meta_info["image_feature"],
        )
        wrench_abs_error = np.abs(pred_wrench - gt_wrench).reshape(
            -1,
            self.model_meta_info["policy"]["args"]["wrench_dim"],
        )
        image_feature_abs_error = np.abs(pred_image_feature - gt_image_feature).reshape(
            -1,
            self.model_meta_info["policy"]["args"]["image_feature_dim"],
        )
        total_abs_error = np.concatenate(
            [normalized_wrench_abs_error, normalized_image_feature_abs_error],
            axis=1,
        )
        return wrench_abs_error, image_feature_abs_error, total_abs_error

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

        wrench_abs_error = np.concatenate(wrench_abs_error_list, axis=0)
        image_feature_abs_error = np.concatenate(
            image_feature_abs_error_list,
            axis=0,
        )
        total_abs_error = np.concatenate(total_abs_error_list, axis=0)
        wrench_dim = self.model_meta_info["policy"]["args"]["wrench_dim"]
        normalized_wrench_abs_error = total_abs_error[:, :wrench_dim]
        normalized_image_feature_abs_error = total_abs_error[:, wrench_dim:]
        wrench_mae = wrench_abs_error.mean(axis=0)

        return {
            "checkpoint": os.path.basename(checkpoint),
            "actual_object_key": actual_object_key,
            "material_object_key": material_object_key,
            "is_correct_material": actual_object_key == material_object_key,
            "episode_count": len(filenames),
            "sample_count": len(wrench_abs_error),
            "SD3 latent mean": image_feature_abs_error.mean(),
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
            "normalized wrench mean": normalized_wrench_abs_error.mean(),
            "normalized total mean": total_abs_error.mean(),
        }

    def print_row(self, row):
        print(
            f"  - ckpt={row['checkpoint']}, "
            f"actual={row['actual_object_key']}, "
            f"PB={row['material_object_key']}, "
            f"correct={row['is_correct_material']}, "
            f"latent={row['SD3 latent mean']:.6f}, "
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
            "SD3 latent mean",
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
                f"{rmb_stem}_hstep_wrench_sd3_latent_loss.png",
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
        material_key_to_latent_mae = {}
        material_key_to_normalized_latent_mae = {}

        for material_object_key in self.material_object_keys:
            material_object_id = self.object_key_to_id[material_object_key]
            gt_wrench_list = []
            pred_wrench_list = []
            latent_mae_list = []
            normalized_latent_mae_list = []

            for batch in dataloader:
                batch = self.move_batch_to_device(batch, material_object_id)
                with torch.inference_mode():
                    pred = self.policy.predict(batch)
                (
                    batch_gt_wrench,
                    batch_pred_wrench,
                    batch_latent_mae,
                    batch_normalized_latent_mae,
                ) = self.compute_hstep_plot_data(batch, pred)
                gt_wrench_list.append(batch_gt_wrench)
                pred_wrench_list.append(batch_pred_wrench)
                latent_mae_list.append(batch_latent_mae)
                normalized_latent_mae_list.append(batch_normalized_latent_mae)

            if gt_wrench is None:
                gt_wrench = np.concatenate(gt_wrench_list, axis=0)
            material_key_to_pred_wrench[material_object_key] = np.concatenate(
                pred_wrench_list,
                axis=0,
            )
            material_key_to_latent_mae[material_object_key] = np.concatenate(
                latent_mae_list,
                axis=0,
            )
            material_key_to_normalized_latent_mae[material_object_key] = np.concatenate(
                normalized_latent_mae_list, axis=0
            )

        return {
            "time_idx": final_time_idx,
            "gt_wrench": gt_wrench,
            "material_key_to_pred_wrench": material_key_to_pred_wrench,
            "material_key_to_latent_mae": material_key_to_latent_mae,
            "material_key_to_normalized_latent_mae": (
                material_key_to_normalized_latent_mae
            ),
        }

    def compute_hstep_plot_data(self, batch, pred):
        gt_wrench_normalized = batch["wrench"][:, -1].detach().cpu().numpy()
        pred_wrench_normalized = pred["wrench"][:, -1].detach().cpu().numpy()
        gt_wrench = denormalize_data(
            gt_wrench_normalized,
            self.model_meta_info["wrench"],
        )
        pred_wrench = denormalize_data(
            pred_wrench_normalized,
            self.model_meta_info["wrench"],
        )

        gt_latent_normalized = batch["image_feature"][:, -1].detach().cpu().numpy()
        pred_latent_normalized = pred["image_feature"][:, -1].detach().cpu().numpy()
        gt_latent = denormalize_data(
            gt_latent_normalized,
            self.model_meta_info["image_feature"],
        )
        pred_latent = denormalize_data(
            pred_latent_normalized,
            self.model_meta_info["image_feature"],
        )
        latent_mae = np.abs(pred_latent - gt_latent).mean(axis=1)
        normalized_latent_mae = np.abs(
            pred_latent_normalized - gt_latent_normalized
        ).mean(axis=1)
        return gt_wrench, pred_wrench, latent_mae, normalized_latent_mae

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
        gt_wrench = plot_data["gt_wrench"]

        for wrench_idx, ax in enumerate(axes[:6]):
            ax.plot(
                time_idx,
                gt_wrench[:, wrench_idx],
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

        ax = axes[6]
        for material_object_key, latent_mae in plot_data[
            "material_key_to_latent_mae"
        ].items():
            ax.plot(
                time_idx,
                latent_mae,
                linewidth=1.2,
                label=f"{material_object_key} PB",
            )
        ax.set_ylabel("SD3 latent MAE")
        ax.grid(True)
        ax.legend(loc="best", fontsize=8)

        ax = axes[7]
        for material_object_key, normalized_latent_mae in plot_data[
            "material_key_to_normalized_latent_mae"
        ].items():
            ax.plot(
                time_idx,
                normalized_latent_mae,
                linewidth=1.2,
                label=f"{material_object_key} PB",
            )
        ax.set_xlabel("skipped time index of t + H - 1")
        ax.set_ylabel("normalized latent MAE")
        ax.grid(True)
        ax.legend(loc="best", fontsize=8)

        fig.suptitle(
            f"{checkpoint_stem} / actual={actual_object_key} / {rmb_stem}",
            fontsize=11,
        )
        fig.tight_layout()
        fig.savefig(output_png)
        plt.close(fig)


if __name__ == "__main__":
    evaluator = EvalWrenchPredictor5SweepDir(**vars(parse_sweep_argument()))
    evaluator.run()
