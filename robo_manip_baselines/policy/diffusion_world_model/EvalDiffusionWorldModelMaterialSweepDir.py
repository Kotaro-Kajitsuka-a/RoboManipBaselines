import csv
import os

import numpy as np
import torch
from tqdm import tqdm

from robo_manip_baselines.common import denormalize_data
from robo_manip_baselines.policy.diffusion_world_model.EvalDiffusionWorldModelSweepCommon import (
    WRENCH_LABELS,
    EvalDiffusionWorldModelSweepBase,
    parse_sweep_argument,
    plt,
)


class EvalDiffusionWorldModelMaterialSweepDir(EvalDiffusionWorldModelSweepBase):
    def get_output_name(self, rmb_dir_name):
        return f"{rmb_dir_name}_world_model_material_sweep"

    def get_output_csv_name(self):
        return "material_sweep_eval.csv"

    def get_summary_csv_name(self):
        return "material_sweep_matrix_summary.csv"

    def get_diagonal_accuracy_csv_name(self):
        return "material_sweep_diagonal_accuracy.csv"

    def get_heatmap_png_name(self, checkpoint_stem):
        return f"{checkpoint_stem}_material_sweep_heatmap.png"

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
        self.reset_sampling_seed()

        for batch in tqdm(
            dataloader,
            desc=f"{os.path.basename(checkpoint)} {actual_object_key} <- {material_object_key}",
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
        image_feature_abs_error = np.concatenate(image_feature_abs_error_list, axis=0)
        total_abs_error = np.concatenate(total_abs_error_list, axis=0)
        wrench_mae = wrench_abs_error.mean(axis=0)
        image_feature_mae = image_feature_abs_error.mean()
        total_mae = total_abs_error.mean()

        return {
            "checkpoint": os.path.basename(checkpoint),
            "actual_object_key": actual_object_key,
            "material_object_key": material_object_key,
            "is_correct_material": actual_object_key == material_object_key,
            "episode_count": len(filenames),
            "sample_count": len(wrench_abs_error),
            "Fx/Fy mean": wrench_mae[:2].mean(),
            "Fx": wrench_mae[0],
            "Fy": wrench_mae[1],
            "Fz": wrench_mae[2],
            "Nx": wrench_mae[3],
            "Ny": wrench_mae[4],
            "Nz": wrench_mae[5],
            "wrench mean": wrench_mae.mean(),
            "image_feature mean": image_feature_mae,
            "total mean": total_mae,
        }

    def compute_abs_error(self, batch, pred):
        start = self.model_meta_info["data"]["n_obs_steps"]
        gt_wrench_normalized = batch["wrench"][:, start:].detach().cpu().numpy()
        pred_wrench_normalized = pred["wrench"][:, start:].detach().cpu().numpy()
        gt_image_feature_normalized = (
            batch["image_feature"][:, start:].detach().cpu().numpy()
        )
        pred_image_feature_normalized = (
            pred["image_feature"][:, start:].detach().cpu().numpy()
        )

        total_abs_error = np.concatenate(
            [
                np.abs(pred_wrench_normalized - gt_wrench_normalized).reshape(
                    -1,
                    self.model_meta_info["policy"]["args"]["wrench_dim"],
                ),
                np.abs(
                    pred_image_feature_normalized - gt_image_feature_normalized
                ).reshape(
                    -1,
                    self.model_meta_info["policy"]["args"]["image_feature_dim"],
                ),
            ],
            axis=1,
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
            -1, self.model_meta_info["policy"]["args"]["wrench_dim"]
        )
        image_feature_abs_error = np.abs(
            pred_image_feature - gt_image_feature
        ).reshape(-1, self.model_meta_info["policy"]["args"]["image_feature_dim"])
        return wrench_abs_error, image_feature_abs_error, total_abs_error

    def print_row(self, row):
        print(
            f"  - ckpt={row['checkpoint']}, "
            f"actual={row['actual_object_key']}, "
            f"material={row['material_object_key']}, "
            f"correct={row['is_correct_material']}, "
            f"Fx/Fy mean={row['Fx/Fy mean']:.6f}, "
            f"wrench mean={row['wrench mean']:.6f}, "
            f"image_feature mean={row['image_feature mean']:.6f}"
        )

    def save_csv(self, rows):
        fieldnames = [
            "checkpoint",
            "actual_object_key",
            "material_object_key",
            "is_correct_material",
            "episode_count",
            "sample_count",
            "Fx/Fy mean",
            "Fx",
            "Fy",
            "Fz",
            "Nx",
            "Ny",
            "Nz",
            "wrench mean",
            "image_feature mean",
            "total mean",
        ]
        with open(self.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def get_heatmap_metrics(self):
        return [
            ("wrench", "wrench mean"),
            ("image feature", "image_feature mean"),
            ("total", "total mean"),
        ]

    def save_episode_plots(self, checkpoint, actual_object_key, filenames):
        checkpoint_stem = os.path.splitext(os.path.basename(checkpoint))[0]
        plot_dir = os.path.join(
            self.episode_dir,
            checkpoint_stem,
            f"actual_{actual_object_key}",
        )
        os.makedirs(plot_dir, exist_ok=True)

        for filename in filenames:
            plot_data = self.evaluate_episode_for_plot(filename)
            rmb_stem = os.path.basename(filename.rstrip("/")).replace(".rmb", "")
            output_png = os.path.join(
                plot_dir,
                f"{rmb_stem}_hstep_fx_fy_image_feature_loss.png",
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
        material_key_to_image_feature_loss = {}

        for material_object_key in self.material_object_keys:
            material_object_id = self.object_key_to_id[material_object_key]
            pred_wrench_list = []
            gt_wrench_list = []
            image_feature_loss_list = []
            self.reset_sampling_seed()

            for batch in dataloader:
                batch = self.move_batch_to_device(batch, material_object_id)
                with torch.inference_mode():
                    pred = self.policy.predict(batch)

                batch_gt_wrench, batch_pred_wrench, batch_image_feature_loss = (
                    self.compute_hstep_plot_data(batch, pred)
                )
                gt_wrench_list.append(batch_gt_wrench)
                pred_wrench_list.append(batch_pred_wrench)
                image_feature_loss_list.append(batch_image_feature_loss)

            if gt_wrench is None:
                gt_wrench = np.concatenate(gt_wrench_list, axis=0)
            material_key_to_pred_wrench[material_object_key] = np.concatenate(
                pred_wrench_list,
                axis=0,
            )
            material_key_to_image_feature_loss[material_object_key] = np.concatenate(
                image_feature_loss_list,
                axis=0,
            )

        return {
            "time_idx": final_time_idx,
            "gt_wrench": gt_wrench,
            "material_key_to_pred_wrench": material_key_to_pred_wrench,
            "material_key_to_image_feature_loss": material_key_to_image_feature_loss,
        }

    def compute_hstep_plot_data(self, batch, pred):
        gt_wrench = batch["wrench"][:, -1].detach().cpu().numpy()
        pred_wrench = pred["wrench"][:, -1].detach().cpu().numpy()
        gt_image_feature = batch["image_feature"][:, -1].detach().cpu().numpy()
        pred_image_feature = pred["image_feature"][:, -1].detach().cpu().numpy()

        gt_wrench = denormalize_data(gt_wrench, self.model_meta_info["wrench"])
        pred_wrench = denormalize_data(pred_wrench, self.model_meta_info["wrench"])
        gt_image_feature = denormalize_data(
            gt_image_feature,
            self.model_meta_info["image_feature"],
        )
        pred_image_feature = denormalize_data(
            pred_image_feature,
            self.model_meta_info["image_feature"],
        )
        image_feature_loss = np.mean(
            np.abs(pred_image_feature - gt_image_feature),
            axis=-1,
        )

        return gt_wrench, pred_wrench, image_feature_loss

    def save_episode_plot(
        self,
        output_png,
        checkpoint_stem,
        actual_object_key,
        rmb_stem,
        plot_data,
    ):
        fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
        time_idx = plot_data["time_idx"]
        gt_wrench = plot_data["gt_wrench"]
        material_key_to_pred_wrench = plot_data["material_key_to_pred_wrench"]
        material_key_to_image_feature_loss = plot_data[
            "material_key_to_image_feature_loss"
        ]

        for wrench_idx, ax in enumerate(axes[:2]):
            ax.plot(
                time_idx,
                gt_wrench[:, wrench_idx],
                color="black",
                linewidth=2.0,
                label=f"GT {WRENCH_LABELS[wrench_idx]}",
            )
            for material_object_key, pred_wrench in material_key_to_pred_wrench.items():
                ax.plot(
                    time_idx,
                    pred_wrench[:, wrench_idx],
                    linewidth=1.2,
                    label=f"{material_object_key} pred",
                )
            ax.set_ylabel(WRENCH_LABELS[wrench_idx])
            ax.grid(True)
            ax.legend(loc="best", fontsize=8)

        ax = axes[2]
        for material_object_key, image_feature_loss in (
            material_key_to_image_feature_loss.items()
        ):
            ax.plot(
                time_idx,
                image_feature_loss,
                linewidth=1.2,
                label=f"{material_object_key} image feature L1",
            )
        ax.set_xlabel("skipped time index of t + H - 1")
        ax.set_ylabel("image feature L1")
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
    evaluator = EvalDiffusionWorldModelMaterialSweepDir(**vars(parse_sweep_argument()))
    evaluator.run()
