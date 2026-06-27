import argparse
import csv
import glob
import os
import pickle
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt

sys.path.append(
    os.path.join(os.path.dirname(__file__), "../../../third_party/diffusion_policy")
)
from robo_manip_baselines.common import (
    DataKey,
    RmbData,
    denormalize_data,
    find_rmb_files,
    get_skipped_data_seq,
)
from robo_manip_baselines.policy.diffusion_world_model.DiffusionWorldModel import (
    DiffusionWorldModel,
)
from robo_manip_baselines.policy.diffusion_world_model.DiffusionWorldModelDataset import (
    DiffusionWorldModelDataset,
)


MATERIAL_OBJECT_KEYS = [
    "WrenchPredObject0",
    "WrenchPredObject1",
    "WrenchPredObject2",
    "WrenchPredObject3",
    "WrenchPredObject4",
]
WRENCH_LABELS = ["Fx", "Fy", "Fz", "Nx", "Ny", "Nz"]


class EvalDiffusionWorldModelDataset(DiffusionWorldModelDataset):
    def __getitem__(self, chunk_idx):
        skip = self.model_meta_info["data"]["skip"]
        horizon = self.model_meta_info["data"]["horizon"]
        episode_idx, start_time_idx = self.chunk_info_list[chunk_idx]
        filename = self.filenames[episode_idx]
        object_id = self.get_object_id(filename)
        clip_info = self.model_meta_info["wrench"]["percentile_clip"]
        source_key = clip_info["source_key"]
        if isinstance(source_key, bytes):
            source_key = source_key.decode()

        with RmbData(filename, self.enable_rmb_cache) as rmb_data:
            episode_len = rmb_data[DataKey.TIME][::skip].shape[0]
            time_idxes = np.clip(
                np.arange(start_time_idx, start_time_idx + horizon), 0, episode_len - 1
            )

            image_feature_key = self.model_meta_info["data"]["image_feature_key"]
            image_feature = get_skipped_data_seq(
                rmb_data[image_feature_key][:], image_feature_key, skip
            )[time_idxes]

            if len(self.model_meta_info["state"]["keys"]) == 0:
                state = np.zeros((horizon, 0), dtype=np.float64)
            else:
                state = np.concatenate(
                    [
                        get_skipped_data_seq(rmb_data[key][:], key, skip)[time_idxes]
                        for key in self.model_meta_info["state"]["keys"]
                    ],
                    axis=1,
                )

            action = np.concatenate(
                [
                    get_skipped_data_seq(rmb_data[key][:], key, skip)[time_idxes]
                    for key in self.model_meta_info["action"]["keys"]
                ],
                axis=1,
            )

            source_wrench = np.asarray(rmb_data[source_key][:])
            clipped_wrench = np.clip(source_wrench, clip_info["min"], clip_info["max"])
            wrench = get_skipped_data_seq(
                clipped_wrench,
                self.model_meta_info["wrench"]["percentile_clip"]["key"],
                skip,
            )[time_idxes]

        state, action, image_feature, wrench = self.pre_convert_data(
            state, action, image_feature, wrench
        )

        return {
            "image_feature": torch.tensor(image_feature, dtype=torch.float32),
            "state": torch.tensor(state, dtype=torch.float32),
            "action": torch.tensor(action, dtype=torch.float32),
            "wrench": torch.tensor(wrench, dtype=torch.float32),
            "object_id": torch.tensor(object_id, dtype=torch.long),
        }


def parse_argument():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "checkpoint_dir",
        type=str,
        help="directory containing model_meta_info.pkl and policy_*.ckpt",
    )
    parser.add_argument(
        "rmb_dir",
        type=str,
        help="validation dataset directory containing RMB episode files",
    )
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument(
        "--num_files",
        type=int,
        default=None,
        help="number of RMB files to evaluate",
    )
    parser.add_argument(
        "--no_plot",
        action="store_true",
        help="disable episode-wise PNG plots",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    return parser.parse_args()


class EvalDiffusionWorldModelMaterialSweepDir:
    def __init__(
        self,
        checkpoint_dir,
        rmb_dir,
        batch_size=64,
        num_files=None,
        no_plot=False,
        seed=42,
    ):
        self.checkpoint_dir = checkpoint_dir
        self.rmb_dir = rmb_dir
        self.batch_size = batch_size
        self.num_files = num_files
        self.no_plot = no_plot
        self.seed = seed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run(self):
        self.setup_model_meta_info()
        self.setup_paths()

        rows = []
        for checkpoint in self.checkpoints:
            print(f"[{self.__class__.__name__}] Evaluate checkpoint: {checkpoint}")
            self.setup_policy(checkpoint)
            for actual_object_key in self.target_object_keys:
                actual_filenames = self.object_key_to_filenames[actual_object_key]
                for material_object_key in MATERIAL_OBJECT_KEYS:
                    row = self.evaluate_object_pair(
                        checkpoint,
                        actual_object_key,
                        material_object_key,
                        actual_filenames,
                    )
                    rows.append(row)
                    self.print_row(row)
                if not self.no_plot:
                    self.save_episode_plots(
                        checkpoint,
                        actual_object_key,
                        actual_filenames,
                    )

        self.save_csv(rows)
        self.save_summary_csv(rows)
        self.save_diagonal_accuracy_csv(rows)
        self.save_heatmaps(rows)
        print(f"[{self.__class__.__name__}] Save csv: {self.output_csv}")

    def setup_model_meta_info(self):
        model_meta_info_path = os.path.join(self.checkpoint_dir, "model_meta_info.pkl")
        with open(model_meta_info_path, "rb") as f:
            self.model_meta_info = pickle.load(f)
        if "percentile_clip" not in self.model_meta_info["wrench"]:
            raise KeyError(
                "[EvalDiffusionWorldModelMaterialSweepDir] "
                "model_meta_info['wrench']['percentile_clip'] is missing. "
                "Please train DiffusionWorldModel with a checkpoint that stores the training clip range."
            )
        print(
            f"[{self.__class__.__name__}] Load model meta info: {model_meta_info_path}"
        )

    def setup_paths(self):
        self.checkpoints = sorted(
            glob.glob(os.path.join(self.checkpoint_dir, "policy_*.ckpt"))
        )
        assert len(self.checkpoints) > 0, self.checkpoint_dir

        rmb_path_list = find_rmb_files(self.rmb_dir, num_files=self.num_files)
        assert len(rmb_path_list) > 0, self.rmb_dir
        self.object_key_to_filenames = self.group_filenames_by_object_key(
            rmb_path_list
        )
        self.target_object_keys = sorted(
            self.object_key_to_filenames,
            key=lambda key: self.object_key_to_id[key],
        )
        assert len(self.target_object_keys) > 0, self.rmb_dir

        rmb_dir_name = os.path.basename(os.path.normpath(self.rmb_dir))
        output_name = f"{rmb_dir_name}_world_model_material_sweep"
        self.output_dir = os.path.join(self.checkpoint_dir, "eval", output_name)
        self.raw_dir = os.path.join(self.output_dir, "raw")
        self.summary_dir = os.path.join(self.output_dir, "summary")
        self.heatmap_dir = os.path.join(self.output_dir, "heatmaps")
        self.episode_dir = os.path.join(self.output_dir, "episodes")
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.summary_dir, exist_ok=True)
        os.makedirs(self.heatmap_dir, exist_ok=True)
        os.makedirs(self.episode_dir, exist_ok=True)
        self.output_csv = os.path.join(self.raw_dir, "material_sweep_eval.csv")
        self.summary_csv = os.path.join(
            self.summary_dir,
            "material_sweep_matrix_summary.csv",
        )
        self.diagonal_accuracy_csv = os.path.join(
            self.summary_dir,
            "material_sweep_diagonal_accuracy.csv",
        )

        print(f"[{self.__class__.__name__}] Device: {self.device}")
        print(f"[{self.__class__.__name__}] Checkpoints: {len(self.checkpoints)}")
        print(f"[{self.__class__.__name__}] RMB files: {len(rmb_path_list)}")
        print(f"[{self.__class__.__name__}] Objects: {self.target_object_keys}")
        print(f"[{self.__class__.__name__}] Material PBs: {MATERIAL_OBJECT_KEYS}")
        print(f"[{self.__class__.__name__}] Output directory: {self.output_dir}")

        for object_key in MATERIAL_OBJECT_KEYS:
            assert object_key in self.object_key_to_id, object_key

    @property
    def object_key_to_id(self):
        return self.model_meta_info["material_property"]["object_key_to_id"]

    def group_filenames_by_object_key(self, filenames):
        object_key_to_filenames = {}
        for filename in filenames:
            object_key = self.extract_object_key(filename)
            if object_key is None:
                continue
            object_key_to_filenames.setdefault(object_key, []).append(filename)
        return object_key_to_filenames

    def extract_object_key(self, filename):
        matched_object_keys = [
            object_key for object_key in self.object_key_to_id if object_key in filename
        ]
        assert len(matched_object_keys) <= 1, (filename, matched_object_keys)
        if len(matched_object_keys) == 0:
            return None
        return matched_object_keys[0]

    def setup_policy(self, checkpoint):
        noise_scheduler = self.construct_noise_scheduler()
        self.policy = DiffusionWorldModel(
            noise_scheduler=noise_scheduler,
            **self.model_meta_info["policy"]["args"],
        )
        self.policy.load_state_dict(
            torch.load(checkpoint, map_location=self.device, weights_only=True)
        )
        self.policy.to(self.device)
        self.policy.eval()

    def construct_noise_scheduler(self):
        scheduler = self.model_meta_info["policy"]["scheduler"]
        if scheduler == "ddpm":
            from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

            return DDPMScheduler(
                **self.model_meta_info["policy"]["noise_scheduler_args"]
            )
        elif scheduler == "ddim":
            from diffusers.schedulers.scheduling_ddim import DDIMScheduler

            return DDIMScheduler(
                **self.model_meta_info["policy"]["noise_scheduler_args"]
            )
        else:
            raise ValueError(f"Invalid scheduler: {scheduler}")

    def evaluate_object_pair(
        self,
        checkpoint,
        actual_object_key,
        material_object_key,
        filenames,
    ):
        dataset = EvalDiffusionWorldModelDataset(
            filenames,
            self.model_meta_info,
            enable_rmb_cache=False,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
        )

        material_object_id = self.object_key_to_id[material_object_key]
        wrench_abs_error_list = []
        image_feature_abs_error_list = []
        total_abs_error_list = []
        torch.manual_seed(self.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.seed)

        for batch in tqdm(
            dataloader,
            desc=f"{os.path.basename(checkpoint)} {actual_object_key} <- {material_object_key}",
            leave=False,
        ):
            batch = {
                key: value.to(self.device) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            batch["object_id"] = torch.full_like(
                batch["object_id"], material_object_id
            )
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
        dataset = EvalDiffusionWorldModelDataset(
            [filename],
            self.model_meta_info,
            enable_rmb_cache=False,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
        )

        final_time_idx = self.get_final_time_idx(filename, dataset)
        gt_wrench = None
        material_key_to_pred_wrench = {}
        material_key_to_image_feature_loss = {}

        for material_object_key in MATERIAL_OBJECT_KEYS:
            material_object_id = self.object_key_to_id[material_object_key]
            pred_wrench_list = []
            gt_wrench_list = []
            image_feature_loss_list = []

            torch.manual_seed(self.seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(self.seed)

            for batch in dataloader:
                batch = {
                    key: value.to(self.device)
                    if isinstance(value, torch.Tensor)
                    else value
                    for key, value in batch.items()
                }
                batch["object_id"] = torch.full_like(
                    batch["object_id"], material_object_id
                )
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

    def get_final_time_idx(self, filename, dataset):
        skip = self.model_meta_info["data"]["skip"]
        horizon = self.model_meta_info["data"]["horizon"]
        with RmbData(filename) as rmb_data:
            episode_len = rmb_data[DataKey.TIME][::skip].shape[0]

        return np.asarray(
            [
                np.clip(start_time_idx + horizon - 1, 0, episode_len - 1)
                for _episode_idx, start_time_idx in dataset.chunk_info_list
            ],
            dtype=np.int64,
        )

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

    def save_summary_csv(self, rows):
        fieldnames = [
            "checkpoint",
            "metric",
            "actual_object_key",
            *MATERIAL_OBJECT_KEYS,
            "best_material_object_key",
            "correct_error",
            "best_error",
            "correct_is_best",
        ]
        summary_rows = []
        for checkpoint in sorted({row["checkpoint"] for row in rows}):
            checkpoint_rows = [
                row for row in rows if row["checkpoint"] == checkpoint
            ]
            for metric_name, row_metric_key in self.get_heatmap_metrics():
                matrix = self.build_error_matrix(checkpoint_rows, row_metric_key)
                for actual_idx, actual_object_key in enumerate(self.target_object_keys):
                    correct_material_idx = MATERIAL_OBJECT_KEYS.index(actual_object_key)
                    best_material_idx = int(np.argmin(matrix[actual_idx]))
                    summary_rows.append(
                        {
                            "checkpoint": checkpoint,
                            "metric": metric_name,
                            "actual_object_key": actual_object_key,
                            **{
                                material_object_key: matrix[
                                    actual_idx,
                                    material_idx,
                                ]
                                for material_idx, material_object_key in enumerate(
                                    MATERIAL_OBJECT_KEYS
                                )
                            },
                            "best_material_object_key": MATERIAL_OBJECT_KEYS[
                                best_material_idx
                            ],
                            "correct_error": matrix[
                                actual_idx,
                                correct_material_idx,
                            ],
                            "best_error": matrix[actual_idx, best_material_idx],
                            "correct_is_best": best_material_idx
                            == correct_material_idx,
                        }
                    )

        with open(self.summary_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"[{self.__class__.__name__}] Save summary csv: {self.summary_csv}")

    def save_diagonal_accuracy_csv(self, rows):
        fieldnames = [
            "checkpoint",
            "metric",
            "diagonal_accuracy",
            "num_correct",
            "num_objects",
        ]
        accuracy_rows = []
        for checkpoint in sorted({row["checkpoint"] for row in rows}):
            checkpoint_rows = [
                row for row in rows if row["checkpoint"] == checkpoint
            ]
            for metric_name, row_metric_key in self.get_heatmap_metrics():
                matrix = self.build_error_matrix(checkpoint_rows, row_metric_key)
                num_correct = 0
                for actual_idx, actual_object_key in enumerate(self.target_object_keys):
                    correct_material_idx = MATERIAL_OBJECT_KEYS.index(actual_object_key)
                    best_material_idx = int(np.argmin(matrix[actual_idx]))
                    if best_material_idx == correct_material_idx:
                        num_correct += 1
                accuracy_rows.append(
                    {
                        "checkpoint": checkpoint,
                        "metric": metric_name,
                        "diagonal_accuracy": num_correct
                        / len(self.target_object_keys),
                        "num_correct": num_correct,
                        "num_objects": len(self.target_object_keys),
                    }
                )

        with open(self.diagonal_accuracy_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(accuracy_rows)
        print(
            f"[{self.__class__.__name__}] Save diagonal accuracy csv: "
            f"{self.diagonal_accuracy_csv}"
        )

    def save_heatmaps(self, rows):
        for checkpoint in sorted({row["checkpoint"] for row in rows}):
            checkpoint_rows = [
                row for row in rows if row["checkpoint"] == checkpoint
            ]
            checkpoint_stem = os.path.splitext(checkpoint)[0]
            output_png = os.path.join(
                self.heatmap_dir,
                f"{checkpoint_stem}_material_sweep_heatmap.png",
            )
            self.save_checkpoint_heatmap(output_png, checkpoint_stem, checkpoint_rows)
            print(f"[{self.__class__.__name__}] Save heatmap: {output_png}")

    def save_checkpoint_heatmap(self, output_png, checkpoint_stem, rows):
        metrics = self.get_heatmap_metrics()
        fig, axes = plt.subplots(len(metrics), 2, figsize=(12, 12))
        for row_idx, (metric_name, row_metric_key) in enumerate(metrics):
            error_matrix = self.build_error_matrix(rows, row_metric_key)
            delta_matrix = self.compute_delta_from_correct(error_matrix)
            self.draw_heatmap(
                axes[row_idx, 0],
                error_matrix,
                f"{metric_name} absolute",
                "viridis",
            )
            self.draw_heatmap(
                axes[row_idx, 1],
                delta_matrix,
                f"{metric_name} delta from correct",
                "coolwarm",
                center_zero=True,
            )

        fig.suptitle(f"{checkpoint_stem} material sweep", fontsize=13)
        fig.tight_layout()
        fig.savefig(output_png)
        plt.close(fig)

    def get_heatmap_metrics(self):
        return [
            ("wrench", "wrench mean"),
            ("image feature", "image_feature mean"),
            ("total", "total mean"),
        ]

    def build_error_matrix(self, rows, metric_key):
        row_by_object_pair = {
            (row["actual_object_key"], row["material_object_key"]): row
            for row in rows
        }
        matrix = np.full(
            (len(self.target_object_keys), len(MATERIAL_OBJECT_KEYS)),
            np.nan,
            dtype=np.float64,
        )
        for actual_idx, actual_object_key in enumerate(self.target_object_keys):
            for material_idx, material_object_key in enumerate(MATERIAL_OBJECT_KEYS):
                matrix[actual_idx, material_idx] = row_by_object_pair[
                    (actual_object_key, material_object_key)
                ][metric_key]
        return matrix

    def compute_delta_from_correct(self, error_matrix):
        delta_matrix = np.zeros_like(error_matrix)
        for actual_idx, actual_object_key in enumerate(self.target_object_keys):
            correct_material_idx = MATERIAL_OBJECT_KEYS.index(actual_object_key)
            delta_matrix[actual_idx] = (
                error_matrix[actual_idx]
                - error_matrix[actual_idx, correct_material_idx]
            )
        return delta_matrix

    def draw_heatmap(
        self,
        ax,
        matrix,
        title,
        cmap,
        center_zero=False,
    ):
        vmin = None
        vmax = None
        if center_zero:
            abs_max = np.nanmax(np.abs(matrix))
            if abs_max == 0:
                abs_max = 1.0
            vmin = -abs_max
            vmax = abs_max
        im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xticks(range(len(MATERIAL_OBJECT_KEYS)))
        ax.set_xticklabels(
            [object_key.replace("WrenchPredObject", "PB") for object_key in MATERIAL_OBJECT_KEYS],
            rotation=45,
            ha="right",
        )
        ax.set_yticks(range(len(self.target_object_keys)))
        ax.set_yticklabels(
            [
                object_key.replace("WrenchPredObject", "Object")
                for object_key in self.target_object_keys
            ]
        )
        ax.set_xlabel("used material PB")
        ax.set_ylabel("actual object")
        for row_idx in range(matrix.shape[0]):
            for col_idx in range(matrix.shape[1]):
                ax.text(
                    col_idx,
                    row_idx,
                    f"{matrix[row_idx, col_idx]:.3g}",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=8,
                )
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


if __name__ == "__main__":
    evaluator = EvalDiffusionWorldModelMaterialSweepDir(**vars(parse_argument()))
    evaluator.run()
