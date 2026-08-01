import csv
import os

import numpy as np
import torch
from tqdm import tqdm

from robo_manip_baselines.common import denormalize_data, get_pose7_from_pose9
from robo_manip_baselines.policy.diffusion_world_model.EvalDiffusionWorldModelMaterialSweepDir import (
    EvalDiffusionWorldModelMaterialSweepDir,
)
from robo_manip_baselines.policy.diffusion_world_model.EvalDiffusionWorldModelSweepCommon import (
    WRENCH_LABELS,
    parse_sweep_argument,
    plt,
)
from robo_manip_baselines.policy.wrench_predictor4.WrenchPredictor4Model import (
    WrenchPredictor4Model,
)


class EvalWrenchPredictor4LiftingSweepDir(EvalDiffusionWorldModelMaterialSweepDir):
    def get_output_name(self, rmb_dir_name):
        return f"{rmb_dir_name}_wrench_predictor4_lifting_sweep"

    def get_output_csv_name(self):
        return "lifting_sweep_eval.csv"

    def get_summary_csv_name(self):
        return "lifting_sweep_matrix_summary.csv"

    def get_diagonal_accuracy_csv_name(self):
        return "lifting_sweep_diagonal_accuracy.csv"

    def get_heatmap_png_name(self, checkpoint_stem):
        return f"{checkpoint_stem}_wrench_predictor4_lifting_sweep_heatmap.png"

    def get_heatmap_metrics(self):
        return [
            ("I-shape block position [m]", "tblock position [m]"),
            ("I-shape block rotation [deg]", "tblock rotation [deg]"),
            ("auxiliary force MAE [N]", "force mean"),
            ("auxiliary torque MAE [N m]", "torque mean"),
            ("normalized total error", "normalized total mean"),
        ]

    def setup_policy(self, checkpoint):
        self.policy = WrenchPredictor4Model(**self.model_meta_info["policy"]["args"])
        self.policy.load_state_dict(
            torch.load(checkpoint, map_location=self.device, weights_only=True)
        )
        self.policy.to(self.device)
        self.policy.eval()

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
        tblock_position_error_list = []
        tblock_rotation_error_list = []
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
            wrench_abs_error, _image_feature_abs_error, total_abs_error = (
                self.compute_abs_error(batch, pred)
            )
            position_error, rotation_error = self.compute_tblock_pose_error(
                batch,
                pred,
            )
            wrench_abs_error_list.append(wrench_abs_error)
            tblock_position_error_list.append(position_error)
            tblock_rotation_error_list.append(rotation_error)
            total_abs_error_list.append(total_abs_error)

        wrench_abs_error = np.concatenate(wrench_abs_error_list, axis=0)
        wrench_mae = wrench_abs_error.mean(axis=0)
        tblock_position_error = np.concatenate(
            tblock_position_error_list,
            axis=0,
        )
        tblock_rotation_error = np.concatenate(
            tblock_rotation_error_list,
            axis=0,
        )
        total_abs_error = np.concatenate(total_abs_error_list, axis=0)

        return {
            "checkpoint": os.path.basename(checkpoint),
            "actual_object_key": actual_object_key,
            "material_object_key": material_object_key,
            "is_correct_material": actual_object_key == material_object_key,
            "episode_count": len(filenames),
            "sample_count": len(wrench_abs_error),
            "tblock position [m]": tblock_position_error.mean(),
            "tblock rotation [deg]": tblock_rotation_error.mean(),
            "Fx": wrench_mae[0],
            "Fy": wrench_mae[1],
            "Fz": wrench_mae[2],
            "Nx": wrench_mae[3],
            "Ny": wrench_mae[4],
            "Nz": wrench_mae[5],
            "force mean": wrench_mae[:3].mean(),
            "torque mean": wrench_mae[3:].mean(),
            "wrench mean": wrench_mae.mean(),
            "normalized total mean": total_abs_error.mean(),
        }

    def compute_tblock_pose_error(self, batch, pred, time_idx=slice(None)):
        start = self.model_meta_info["data"]["n_obs_steps"]
        gt_pose_normalized = (
            batch["image_feature"][:, start:, :][:, time_idx].detach().cpu().numpy()
        )
        pred_pose_normalized = (
            pred["image_feature"][:, start:, :][:, time_idx].detach().cpu().numpy()
        )
        gt_pose = denormalize_data(
            gt_pose_normalized,
            self.model_meta_info["image_feature"],
        )
        pred_pose = denormalize_data(
            pred_pose_normalized,
            self.model_meta_info["image_feature"],
        )
        assert gt_pose.shape[-1] == 9, gt_pose.shape
        gt_pose = get_pose7_from_pose9(gt_pose)
        pred_pose = get_pose7_from_pose9(pred_pose)

        position_error = np.linalg.norm(
            pred_pose[..., :3] - gt_pose[..., :3],
            axis=-1,
        )
        gt_quaternion = self.normalize_quaternion(gt_pose[..., 3:])
        pred_quaternion = self.normalize_quaternion(pred_pose[..., 3:])
        quaternion_dot = np.abs(np.sum(gt_quaternion * pred_quaternion, axis=-1))
        rotation_error = np.degrees(2.0 * np.arccos(np.clip(quaternion_dot, 0.0, 1.0)))
        return position_error.reshape(-1), rotation_error.reshape(-1)

    @staticmethod
    def normalize_quaternion(quaternion):
        norm = np.linalg.norm(quaternion, axis=-1, keepdims=True)
        return quaternion / np.clip(norm, 1e-12, None)

    def print_row(self, row):
        print(
            f"  - ckpt={row['checkpoint']}, "
            f"actual={row['actual_object_key']}, "
            f"PB={row['material_object_key']}, "
            f"correct={row['is_correct_material']}, "
            f"position={row['tblock position [m]']:.6f} m, "
            f"rotation={row['tblock rotation [deg]']:.3f} deg, "
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
            "tblock position [m]",
            "tblock rotation [deg]",
            "Fx",
            "Fy",
            "Fz",
            "Nx",
            "Ny",
            "Nz",
            "force mean",
            "torque mean",
            "wrench mean",
            "normalized total mean",
        ]
        with open(self.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
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

        for filename in filenames:
            plot_data = self.evaluate_episode_for_plot(filename)
            rmb_stem = os.path.basename(filename.rstrip("/")).replace(".rmb", "")
            output_png = os.path.join(
                plot_dir,
                f"{rmb_stem}_hstep_wrench_tblock_pose_loss.png",
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
        material_key_to_position_error = {}
        material_key_to_rotation_error = {}

        for material_object_key in self.material_object_keys:
            material_object_id = self.object_key_to_id[material_object_key]
            gt_wrench_list = []
            pred_wrench_list = []
            position_error_list = []
            rotation_error_list = []

            for batch in dataloader:
                batch = self.move_batch_to_device(batch, material_object_id)
                with torch.inference_mode():
                    pred = self.policy.predict(batch)
                (
                    batch_gt_wrench,
                    batch_pred_wrench,
                    batch_position_error,
                    batch_rotation_error,
                ) = self.compute_hstep_plot_data(batch, pred)
                gt_wrench_list.append(batch_gt_wrench)
                pred_wrench_list.append(batch_pred_wrench)
                position_error_list.append(batch_position_error)
                rotation_error_list.append(batch_rotation_error)

            if gt_wrench is None:
                gt_wrench = np.concatenate(gt_wrench_list, axis=0)
            material_key_to_pred_wrench[material_object_key] = np.concatenate(
                pred_wrench_list,
                axis=0,
            )
            material_key_to_position_error[material_object_key] = np.concatenate(
                position_error_list,
                axis=0,
            )
            material_key_to_rotation_error[material_object_key] = np.concatenate(
                rotation_error_list,
                axis=0,
            )

        return {
            "time_idx": final_time_idx,
            "gt_wrench": gt_wrench,
            "material_key_to_pred_wrench": material_key_to_pred_wrench,
            "material_key_to_position_error": material_key_to_position_error,
            "material_key_to_rotation_error": material_key_to_rotation_error,
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
        position_error, rotation_error = self.compute_tblock_pose_error(
            batch,
            pred,
            time_idx=-1,
        )
        return gt_wrench, pred_wrench, position_error, rotation_error

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
        material_key_to_pred_wrench = plot_data["material_key_to_pred_wrench"]
        material_key_to_position_error = plot_data["material_key_to_position_error"]
        material_key_to_rotation_error = plot_data["material_key_to_rotation_error"]

        for wrench_idx, ax in enumerate(axes[:6]):
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
                    label=f"{material_object_key} PB",
                )
            ax.set_ylabel(WRENCH_LABELS[wrench_idx])
            ax.grid(True)
            ax.legend(loc="best", fontsize=8)

        ax = axes[6]
        for (
            material_object_key,
            position_error,
        ) in material_key_to_position_error.items():
            ax.plot(
                time_idx,
                100.0 * position_error,
                linewidth=1.2,
                label=f"{material_object_key} PB",
            )
        ax.set_ylabel("block position [cm]")
        ax.grid(True)
        ax.legend(loc="best", fontsize=8)

        ax = axes[7]
        for (
            material_object_key,
            rotation_error,
        ) in material_key_to_rotation_error.items():
            ax.plot(
                time_idx,
                rotation_error,
                linewidth=1.2,
                label=f"{material_object_key} PB",
            )
        ax.set_xlabel("skipped time index of t + H - 1")
        ax.set_ylabel("block rotation [deg]")
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
    evaluator = EvalWrenchPredictor4LiftingSweepDir(**vars(parse_sweep_argument()))
    evaluator.run()
