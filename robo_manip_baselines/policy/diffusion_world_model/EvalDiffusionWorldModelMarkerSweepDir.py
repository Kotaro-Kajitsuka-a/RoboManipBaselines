import csv
import os

import numpy as np
import torch
from tqdm import tqdm

from robo_manip_baselines.common import denormalize_data
from robo_manip_baselines.policy.diffusion_world_model.EvalDiffusionWorldModelSweepCommon import (
    MATERIAL_OBJECT_KEYS,
    EvalDiffusionWorldModelSweepBase,
    parse_sweep_argument,
    plt,
)


class EvalDiffusionWorldModelMarkerSweepDir(EvalDiffusionWorldModelSweepBase):
    MARKER_DIM = 9

    def setup_model_meta_info(self):
        super().setup_model_meta_info()
        image_feature_key = self.model_meta_info["data"]["image_feature_key"]
        image_feature_dim = self.model_meta_info["policy"]["args"][
            "image_feature_dim"
        ]
        if not image_feature_key.endswith("_apriltag_pose_xy_axis"):
            raise ValueError(
                f"[{self.__class__.__name__}] image_feature_key must be an AprilTag pose key: "
                f"{image_feature_key}"
            )
        if image_feature_dim != self.MARKER_DIM:
            raise ValueError(
                f"[{self.__class__.__name__}] marker feature dim must be {self.MARKER_DIM}: "
                f"{image_feature_dim}"
            )

    def get_output_name(self, rmb_dir_name):
        return f"{rmb_dir_name}_world_model_marker_sweep"

    def get_output_csv_name(self):
        return "marker_sweep_eval.csv"

    def get_summary_csv_name(self):
        return "marker_sweep_matrix_summary.csv"

    def get_diagonal_accuracy_csv_name(self):
        return "marker_sweep_diagonal_accuracy.csv"

    def get_heatmap_png_name(self, checkpoint_stem):
        return f"{checkpoint_stem}_marker_sweep_heatmap.png"

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
        marker_xyz_l2_list = []
        marker_yaw_z_abs_deg_list = []
        marker_feature_l1_list = []
        total_normalized_abs_error_list = []
        self.reset_sampling_seed()

        for batch in tqdm(
            dataloader,
            desc=f"{os.path.basename(checkpoint)} {actual_object_key} <- {material_object_key}",
            leave=False,
        ):
            batch = self.move_batch_to_device(batch, material_object_id)
            with torch.inference_mode():
                pred = self.policy.predict(batch)
            marker_error = self.compute_marker_error(batch, pred)
            wrench_abs_error_list.append(marker_error["wrench_abs_error"])
            marker_xyz_l2_list.append(marker_error["marker_xyz_l2"])
            marker_yaw_z_abs_deg_list.append(marker_error["marker_yaw_z_abs_deg"])
            marker_feature_l1_list.append(marker_error["marker_feature_l1"])
            total_normalized_abs_error_list.append(
                marker_error["total_normalized_abs_error"]
            )

        wrench_abs_error = np.concatenate(wrench_abs_error_list, axis=0)
        wrench_mae = wrench_abs_error.mean(axis=0)
        marker_xyz_l2 = np.concatenate(marker_xyz_l2_list, axis=0)
        marker_yaw_z_abs_deg = np.concatenate(marker_yaw_z_abs_deg_list, axis=0)
        marker_feature_l1 = np.concatenate(marker_feature_l1_list, axis=0)
        total_normalized_abs_error = np.concatenate(
            total_normalized_abs_error_list,
            axis=0,
        )

        return {
            "checkpoint": os.path.basename(checkpoint),
            "actual_object_key": actual_object_key,
            "material_object_key": material_object_key,
            "is_correct_material": actual_object_key == material_object_key,
            "episode_count": len(filenames),
            "sample_count": len(marker_xyz_l2),
            "Fx/Fy mean": wrench_mae[:2].mean(),
            "Fx": wrench_mae[0],
            "Fy": wrench_mae[1],
            "Fz": wrench_mae[2],
            "Nx": wrench_mae[3],
            "Ny": wrench_mae[4],
            "Nz": wrench_mae[5],
            "wrench mean": wrench_mae.mean(),
            "marker xyz l2 [m]": marker_xyz_l2.mean(),
            "marker yaw z abs [deg]": marker_yaw_z_abs_deg.mean(),
            "marker feature L1": marker_feature_l1.mean(),
            "total normalized mean": total_normalized_abs_error.mean(),
        }

    def compute_marker_error(self, batch, pred):
        start = self.model_meta_info["data"]["n_obs_steps"]
        gt_wrench_normalized = batch["wrench"][:, start:].detach().cpu().numpy()
        pred_wrench_normalized = pred["wrench"][:, start:].detach().cpu().numpy()
        gt_marker_normalized = batch["image_feature"][:, start:].detach().cpu().numpy()
        pred_marker_normalized = pred["image_feature"][:, start:].detach().cpu().numpy()

        total_normalized_abs_error = np.concatenate(
            [
                np.abs(pred_wrench_normalized - gt_wrench_normalized).reshape(
                    -1,
                    self.model_meta_info["policy"]["args"]["wrench_dim"],
                ),
                np.abs(pred_marker_normalized - gt_marker_normalized).reshape(
                    -1,
                    self.MARKER_DIM,
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
        gt_marker = denormalize_data(
            gt_marker_normalized,
            self.model_meta_info["image_feature"],
        ).reshape(-1, self.MARKER_DIM)
        pred_marker = denormalize_data(
            pred_marker_normalized,
            self.model_meta_info["image_feature"],
        ).reshape(-1, self.MARKER_DIM)

        wrench_abs_error = np.abs(pred_wrench - gt_wrench).reshape(
            -1, self.model_meta_info["policy"]["args"]["wrench_dim"]
        )
        marker_xyz_l2 = np.linalg.norm(pred_marker[:, :3] - gt_marker[:, :3], axis=1)
        marker_yaw_z_abs_deg = np.rad2deg(
            np.abs(
                self.wrap_angle(
                    self.get_marker_yaw_z(pred_marker) - self.get_marker_yaw_z(gt_marker)
                )
            )
        )
        marker_feature_l1 = np.mean(np.abs(pred_marker - gt_marker), axis=1)

        return {
            "wrench_abs_error": wrench_abs_error,
            "marker_xyz_l2": marker_xyz_l2,
            "marker_yaw_z_abs_deg": marker_yaw_z_abs_deg,
            "marker_feature_l1": marker_feature_l1,
            "total_normalized_abs_error": total_normalized_abs_error,
        }

    def get_marker_yaw_z(self, marker):
        x_axis = marker[:, 3:6]
        return np.arctan2(x_axis[:, 1], x_axis[:, 0])

    def wrap_angle(self, angle):
        return np.arctan2(np.sin(angle), np.cos(angle))

    def print_row(self, row):
        print(
            f"  - ckpt={row['checkpoint']}, "
            f"actual={row['actual_object_key']}, "
            f"material={row['material_object_key']}, "
            f"correct={row['is_correct_material']}, "
            f"marker xyz={row['marker xyz l2 [m]']:.6f} m, "
            f"marker yaw z={row['marker yaw z abs [deg]']:.3f} deg, "
            f"wrench mean={row['wrench mean']:.6f}"
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
            "marker xyz l2 [m]",
            "marker yaw z abs [deg]",
            "marker feature L1",
            "total normalized mean",
        ]
        with open(self.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def get_heatmap_metrics(self):
        return [
            ("wrench", "wrench mean"),
            ("marker xyz [m]", "marker xyz l2 [m]"),
            ("marker yaw z [deg]", "marker yaw z abs [deg]"),
            ("marker feature L1", "marker feature L1"),
            ("total normalized", "total normalized mean"),
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
            plot_data = self.evaluate_marker_episode_for_plot(filename)
            rmb_stem = os.path.basename(filename.rstrip("/")).replace(".rmb", "")
            output_png = os.path.join(
                plot_dir,
                f"{rmb_stem}_hstep_marker_pose.png",
            )
            self.save_marker_episode_plot(
                output_png,
                checkpoint_stem,
                actual_object_key,
                rmb_stem,
                plot_data,
            )

    def evaluate_marker_episode_for_plot(self, filename):
        dataset, dataloader = self.make_dataloader([filename])

        final_time_idx = self.get_final_time_idx(filename, dataset)
        gt_marker = None
        material_key_to_pred_marker = {}

        for material_object_key in MATERIAL_OBJECT_KEYS:
            material_object_id = self.object_key_to_id[material_object_key]
            pred_marker_list = []
            gt_marker_list = []
            self.reset_sampling_seed()

            for batch in dataloader:
                batch = self.move_batch_to_device(batch, material_object_id)
                with torch.inference_mode():
                    pred = self.policy.predict(batch)

                batch_gt_marker, batch_pred_marker = self.compute_hstep_marker_plot_data(
                    batch,
                    pred,
                )
                gt_marker_list.append(batch_gt_marker)
                pred_marker_list.append(batch_pred_marker)

            if gt_marker is None:
                gt_marker = np.concatenate(gt_marker_list, axis=0)
            material_key_to_pred_marker[material_object_key] = np.concatenate(
                pred_marker_list,
                axis=0,
            )

        return {
            "time_idx": final_time_idx,
            "gt_marker": gt_marker,
            "material_key_to_pred_marker": material_key_to_pred_marker,
        }

    def compute_hstep_marker_plot_data(self, batch, pred):
        gt_marker = batch["image_feature"][:, -1].detach().cpu().numpy()
        pred_marker = pred["image_feature"][:, -1].detach().cpu().numpy()

        gt_marker = denormalize_data(
            gt_marker,
            self.model_meta_info["image_feature"],
        )
        pred_marker = denormalize_data(
            pred_marker,
            self.model_meta_info["image_feature"],
        )
        return gt_marker, pred_marker

    def save_marker_episode_plot(
        self,
        output_png,
        checkpoint_stem,
        actual_object_key,
        rmb_stem,
        plot_data,
    ):
        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        time_idx = plot_data["time_idx"]
        gt_marker = plot_data["gt_marker"]
        material_key_to_pred_marker = plot_data["material_key_to_pred_marker"]
        gt_yaw_z_deg = np.rad2deg(self.get_marker_yaw_z(gt_marker))

        plot_specs = [
            ("marker x [m]", lambda marker: marker[:, 0]),
            ("marker y [m]", lambda marker: marker[:, 1]),
            ("marker z [m]", lambda marker: marker[:, 2]),
            (
                "marker yaw z [deg]",
                lambda marker: np.rad2deg(self.get_marker_yaw_z(marker)),
            ),
        ]
        for ax, (ylabel, getter) in zip(axes, plot_specs):
            if ylabel == "marker yaw z [deg]":
                gt_value = gt_yaw_z_deg
            else:
                gt_value = getter(gt_marker)
            ax.plot(
                time_idx,
                gt_value,
                color="black",
                linewidth=2.0,
                label=f"GT {ylabel}",
            )
            for material_object_key, pred_marker in material_key_to_pred_marker.items():
                ax.plot(
                    time_idx,
                    getter(pred_marker),
                    linewidth=1.2,
                    label=f"{material_object_key} pred",
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
    evaluator = EvalDiffusionWorldModelMarkerSweepDir(**vars(parse_sweep_argument()))
    evaluator.run()
