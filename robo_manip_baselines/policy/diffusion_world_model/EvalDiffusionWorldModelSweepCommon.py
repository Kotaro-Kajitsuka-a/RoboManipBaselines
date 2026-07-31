import argparse
import csv
import glob
import os
import pickle
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt

sys.path.append(
    os.path.join(os.path.dirname(__file__), "../../../third_party/diffusion_policy")
)
from robo_manip_baselines.common import (
    DataKey,
    RmbData,
    convert_data_to_policy,
    find_rmb_files,
    get_skipped_data_seq,
)
from robo_manip_baselines.policy.diffusion_world_model.DiffusionWorldModel import (
    DiffusionWorldModel,
)
from robo_manip_baselines.policy.diffusion_world_model.DiffusionWorldModelDataset import (
    DiffusionWorldModelDataset,
)


DEFAULT_MAX_MATERIAL_OBJECT_ID = 4
WRENCH_LABELS = ["Fx", "Fy", "Fz", "Nx", "Ny", "Nz"]


def parse_sweep_argument():
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
    parser.add_argument(
        "--max_material_object_id",
        type=int,
        default=DEFAULT_MAX_MATERIAL_OBJECT_ID,
        help="maximum WrenchPredObject id used as material PB sweep targets",
    )
    parser.add_argument(
        "--plot_time_offsets",
        type=int,
        nargs="*",
        default=None,
        help="target offsets used for episode plots and videos; default is horizon - 1",
    )
    parser.add_argument(
        "--checkpoint_names",
        type=str,
        nargs="*",
        default=None,
        help="checkpoint basenames to evaluate, e.g. policy_best.ckpt policy_last.ckpt",
    )
    parser.add_argument(
        "--output_suffix",
        type=str,
        default="",
        help="suffix appended to the evaluation output directory name",
    )
    return parser.parse_args()


class EvalDiffusionWorldModelDataset(DiffusionWorldModelDataset):
    def __getitem__(self, chunk_idx):
        skip = self.model_meta_info["data"]["skip"]
        horizon = self.model_meta_info["data"]["horizon"]
        episode_idx, start_time_idx = self.chunk_info_list[chunk_idx]
        filename = self.filenames[episode_idx]
        object_id = self.get_object_id(filename)

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
                        convert_data_to_policy(
                            get_skipped_data_seq(rmb_data[key][:], key, skip)[
                                time_idxes
                            ],
                            key,
                        )
                        for key in self.model_meta_info["state"]["keys"]
                    ],
                    axis=1,
                )

            action = np.concatenate(
                [
                    convert_data_to_policy(
                        get_skipped_data_seq(rmb_data[key][:], key, skip)[time_idxes],
                        key,
                    )
                    for key in self.model_meta_info["action"]["keys"]
                ],
                axis=1,
            )

            wrench = self.load_wrench(rmb_data, skip)[time_idxes]

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

    def load_wrench(self, rmb_data, skip):
        clip_info = self.model_meta_info["wrench"]["percentile_clip"]
        source_key = clip_info["source_key"]
        if isinstance(source_key, bytes):
            source_key = source_key.decode()
        source_wrench = np.asarray(rmb_data[source_key][:])
        clipped_wrench = np.clip(
            source_wrench,
            clip_info["min"],
            clip_info["max"],
        )
        return get_skipped_data_seq(clipped_wrench, clip_info["key"], skip)


class EvalDiffusionWorldModelSweepBase:
    def __init__(
        self,
        checkpoint_dir,
        rmb_dir,
        batch_size=64,
        num_files=None,
        no_plot=False,
        seed=42,
        max_material_object_id=DEFAULT_MAX_MATERIAL_OBJECT_ID,
        plot_time_offsets=None,
        checkpoint_names=None,
        output_suffix="",
    ):
        self.checkpoint_dir = checkpoint_dir
        self.rmb_dir = rmb_dir
        self.batch_size = batch_size
        self.num_files = num_files
        self.no_plot = no_plot
        self.seed = seed
        self.plot_time_offsets = plot_time_offsets
        self.checkpoint_names = checkpoint_names
        self.output_suffix = output_suffix
        assert max_material_object_id >= 0, max_material_object_id
        self.material_object_keys = [
            f"WrenchPredObject{object_id}"
            for object_id in range(max_material_object_id + 1)
        ]
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
                for material_object_key in self.material_object_keys:
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
                f"[{self.__class__.__name__}] "
                "model_meta_info['wrench']['percentile_clip'] is missing. "
                "Please train DiffusionWorldModel with --wrench_source_key."
            )
        print(
            f"[{self.__class__.__name__}] Load model meta info: {model_meta_info_path}"
        )

    def setup_paths(self):
        self.checkpoints = sorted(
            glob.glob(os.path.join(self.checkpoint_dir, "policy_*.ckpt"))
        )
        if self.checkpoint_names is not None:
            checkpoint_name_set = set(self.checkpoint_names)
            self.checkpoints = [
                checkpoint
                for checkpoint in self.checkpoints
                if os.path.basename(checkpoint) in checkpoint_name_set
            ]
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
        self.output_dir = os.path.join(
            self.checkpoint_dir,
            "eval",
            self.get_output_name(rmb_dir_name) + self.output_suffix,
        )
        self.raw_dir = os.path.join(self.output_dir, "raw")
        self.summary_dir = os.path.join(self.output_dir, "summary")
        self.heatmap_dir = os.path.join(self.output_dir, "heatmaps")
        self.episode_dir = os.path.join(self.output_dir, "episodes")
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.summary_dir, exist_ok=True)
        os.makedirs(self.heatmap_dir, exist_ok=True)
        os.makedirs(self.episode_dir, exist_ok=True)
        self.output_csv = os.path.join(self.raw_dir, self.get_output_csv_name())
        self.summary_csv = os.path.join(
            self.summary_dir,
            self.get_summary_csv_name(),
        )
        self.diagonal_accuracy_csv = os.path.join(
            self.summary_dir,
            self.get_diagonal_accuracy_csv_name(),
        )

        print(f"[{self.__class__.__name__}] Device: {self.device}")
        print(f"[{self.__class__.__name__}] Checkpoints: {len(self.checkpoints)}")
        print(f"[{self.__class__.__name__}] RMB files: {len(rmb_path_list)}")
        print(f"[{self.__class__.__name__}] Objects: {self.target_object_keys}")
        print(f"[{self.__class__.__name__}] Material PBs: {self.material_object_keys}")
        print(f"[{self.__class__.__name__}] Output directory: {self.output_dir}")

        for object_key in self.material_object_keys:
            assert object_key in self.object_key_to_id, object_key
        for object_key in self.target_object_keys:
            assert object_key in self.material_object_keys, (
                object_key,
                self.material_object_keys,
            )

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

    def make_dataloader(self, filenames):
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
        return dataset, dataloader

    def move_batch_to_device(self, batch, material_object_id):
        batch = {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        batch["object_id"] = torch.full_like(batch["object_id"], material_object_id)
        return batch

    def reset_sampling_seed(self):
        torch.manual_seed(self.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.seed)

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

    def save_summary_csv(self, rows):
        fieldnames = [
            "checkpoint",
            "metric",
            "actual_object_key",
            *self.material_object_keys,
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
                    correct_material_idx = self.material_object_keys.index(
                        actual_object_key
                    )
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
                                    self.material_object_keys
                                )
                            },
                            "best_material_object_key": self.material_object_keys[
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
                    correct_material_idx = self.material_object_keys.index(
                        actual_object_key
                    )
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
                self.get_heatmap_png_name(checkpoint_stem),
            )
            self.save_checkpoint_heatmap(output_png, checkpoint_stem, checkpoint_rows)
            print(f"[{self.__class__.__name__}] Save heatmap: {output_png}")

    def save_checkpoint_heatmap(self, output_png, checkpoint_stem, rows):
        metrics = self.get_heatmap_metrics()
        fig, axes = plt.subplots(len(metrics), 2, figsize=(12, 3.5 * len(metrics)))
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

        fig.suptitle(f"{checkpoint_stem} sweep", fontsize=13)
        fig.tight_layout()
        fig.savefig(output_png)
        plt.close(fig)

    def build_error_matrix(self, rows, metric_key):
        row_by_object_pair = {
            (row["actual_object_key"], row["material_object_key"]): row
            for row in rows
        }
        matrix = np.full(
            (len(self.target_object_keys), len(self.material_object_keys)),
            np.nan,
            dtype=np.float64,
        )
        for actual_idx, actual_object_key in enumerate(self.target_object_keys):
            for material_idx, material_object_key in enumerate(
                self.material_object_keys
            ):
                matrix[actual_idx, material_idx] = row_by_object_pair[
                    (actual_object_key, material_object_key)
                ][metric_key]
        return matrix

    def compute_delta_from_correct(self, error_matrix):
        delta_matrix = np.zeros_like(error_matrix)
        for actual_idx, actual_object_key in enumerate(self.target_object_keys):
            correct_material_idx = self.material_object_keys.index(actual_object_key)
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
        ax.set_xticks(range(len(self.material_object_keys)))
        ax.set_xticklabels(
            [
                object_key.replace("WrenchPredObject", "PB")
                for object_key in self.material_object_keys
            ],
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
