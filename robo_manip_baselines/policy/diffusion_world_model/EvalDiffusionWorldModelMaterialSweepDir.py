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

            image_feature = get_skipped_data_seq(
                rmb_data[self.IMAGE_FEATURE_KEY][:], self.IMAGE_FEATURE_KEY, skip
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
        os.makedirs(self.output_dir, exist_ok=True)
        self.output_csv = os.path.join(self.output_dir, f"{output_name}.csv")

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
            wrench_abs_error, image_feature_abs_error = self.compute_abs_error(
                batch, pred
            )
            wrench_abs_error_list.append(wrench_abs_error)
            image_feature_abs_error_list.append(image_feature_abs_error)

        wrench_abs_error = np.concatenate(wrench_abs_error_list, axis=0)
        image_feature_abs_error = np.concatenate(image_feature_abs_error_list, axis=0)
        wrench_mae = wrench_abs_error.mean(axis=0)
        image_feature_mae = image_feature_abs_error.mean()

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
        }

    def save_episode_plots(self, checkpoint, actual_object_key, filenames):
        checkpoint_stem = os.path.splitext(os.path.basename(checkpoint))[0]
        plot_dir = os.path.join(
            self.output_dir,
            "plots",
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
        gt_wrench = batch["wrench"][:, start:].detach().cpu().numpy()
        pred_wrench = pred["wrench"][:, start:].detach().cpu().numpy()
        gt_image_feature = batch["image_feature"][:, start:].detach().cpu().numpy()
        pred_image_feature = pred["image_feature"][:, start:].detach().cpu().numpy()

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

        wrench_abs_error = np.abs(pred_wrench - gt_wrench).reshape(
            -1, self.model_meta_info["policy"]["args"]["wrench_dim"]
        )
        image_feature_abs_error = np.abs(
            pred_image_feature - gt_image_feature
        ).reshape(-1, self.model_meta_info["policy"]["args"]["image_feature_dim"])
        return wrench_abs_error, image_feature_abs_error

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
        ]
        with open(self.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    evaluator = EvalDiffusionWorldModelMaterialSweepDir(**vars(parse_argument()))
    evaluator.run()
