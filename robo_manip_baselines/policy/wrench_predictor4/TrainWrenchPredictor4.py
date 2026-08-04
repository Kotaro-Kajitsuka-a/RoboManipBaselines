import argparse
import copy
import math
import os

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from robo_manip_baselines.common import (
    DataKey,
    RmbData,
    TrainBase,
    convert_data_to_policy,
    find_rmb_files,
    get_skipped_data_seq,
)
from robo_manip_baselines.misc.AddPercentileClippedWrenchToRmbData import (
    AddPercentileClippedWrenchToRmbData,
    get_percentile_clip_wrench_key,
)

from .WrenchPredictor4Dataset import WrenchPredictor4Dataset
from .WrenchPredictor4Model import WrenchPredictor4Model


def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps,
    num_training_steps,
):
    assert num_warmup_steps >= 0, num_warmup_steps
    assert num_training_steps > 0, num_training_steps

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / max(1, num_warmup_steps)
        progress = float(current_step - num_warmup_steps) / max(
            1,
            num_training_steps - num_warmup_steps,
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class TrainWrenchPredictor4(TrainBase):
    DatasetClass = WrenchPredictor4Dataset

    def setup_args(self):
        super().setup_args()
        if self.args.backbone != "transformer":
            raise ValueError(
                f"[{self.__class__.__name__}] WrenchPredictor4 uses only transformer backbone."
            )
        if self.args.scheduler != "ddpm":
            raise ValueError(
                f"[{self.__class__.__name__}] WrenchPredictor4 accepts only the "
                "legacy --scheduler ddpm setting."
            )
        if self.args.use_ema:
            raise ValueError(
                f"[{self.__class__.__name__}] WrenchPredictor4 does not use EMA."
            )

    def set_additional_args(self, parser):
        parser.set_defaults(
            enable_rmb_cache=True,
            norm_type="limits",
            batch_size=64,
            num_epochs=200,
            lr=1e-4,
        )
        parser.add_argument(
            "--weight_decay", type=float, default=1e-6, help="weight decay"
        )
        parser.add_argument(
            "--use_ema",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="legacy compatibility option; WrenchPredictor4 does not use EMA",
        )
        parser.add_argument(
            "--backbone",
            type=str,
            default="transformer",
            choices=["transformer"],
            help="legacy compatibility option; WrenchPredictor4 uses a Transformer",
        )
        parser.add_argument(
            "--scheduler",
            type=str,
            default="ddpm",
            choices=["ddpm"],
            help="legacy compatibility option retained for existing commands",
        )
        parser.add_argument(
            "--horizon", type=int, default=16, help="prediction horizon"
        )
        parser.add_argument(
            "--n_obs_steps",
            type=int,
            default=2,
            help="number of image_feature/state steps used as condition",
        )
        parser.add_argument(
            "--image_feature_key",
            type=str,
            required=True,
            help="RMB key used as image feature target and condition",
        )
        parser.add_argument(
            "--wrench_source_key",
            type=str,
            required=True,
            choices=[
                DataKey.MEASURED_EEF_WRENCH,
                DataKey.MEASURED_EEF_WRENCH_MOVING_AVERAGE,
            ],
            help="RMB source wrench key to percentile-clip and use as prediction target",
        )
        parser.add_argument(
            "--pb_dim",
            type=int,
            default=9,
            help="dimension of object-wise material property vector",
        )
        parser.add_argument(
            "--hidden_dim",
            type=int,
            default=256,
            help="Transformer hidden dimension",
        )
        parser.add_argument(
            "--nhead",
            type=int,
            default=8,
            help="number of Transformer attention heads",
        )
        parser.add_argument(
            "--num_encoder_layers",
            type=int,
            default=4,
            help="number of Transformer encoder layers",
        )
        parser.add_argument(
            "--num_decoder_layers",
            type=int,
            default=2,
            help="number of Transformer decoder layers",
        )
        parser.add_argument(
            "--dim_feedforward",
            type=int,
            default=1024,
            help="Transformer feedforward dimension",
        )
        parser.add_argument(
            "--dropout",
            type=float,
            default=0.1,
            help="Transformer dropout",
        )
        parser.add_argument(
            "--output_head",
            type=str,
            default="decoder",
            choices=["decoder", "mlp", "mlp_only"],
            help="output head type",
        )
        parser.add_argument(
            "--val_dataset_dir",
            type=str,
            default=None,
            help="separate validation dataset directory",
        )
        parser.add_argument(
            "--wrench_loss_weight",
            type=float,
            default=1.0,
            help="weight of the auxiliary wrench prediction loss",
        )

    def setup_rmb_files(self):
        super().setup_rmb_files()
        if self.args.val_dataset_dir is None:
            return

        self.val_filenames = find_rmb_files(self.args.val_dataset_dir)
        train_filenames = {os.path.realpath(path) for path in self.all_filenames}
        val_filenames = {os.path.realpath(path) for path in self.val_filenames}
        assert train_filenames.isdisjoint(val_filenames)

    def setup_model_meta_info(self):
        super().setup_model_meta_info()
        self.model_meta_info["data"].update(
            {
                "horizon": self.args.horizon,
                "n_obs_steps": self.args.n_obs_steps,
                "n_action_steps": 1,
                "image_feature_key": self.args.image_feature_key,
            }
        )
        self.model_meta_info["wrench"] = {
            "key": get_percentile_clip_wrench_key(self.args.wrench_source_key),
            "source_key": self.args.wrench_source_key,
        }
        self.model_meta_info["material_property"] = {
            "pb_dim": self.args.pb_dim,
            "object_key_to_id": WrenchPredictor4Dataset.OBJECT_KEY_TO_ID,
        }
        self.model_meta_info["policy"].update(
            {
                "use_ema": self.args.use_ema,
                "backbone": self.args.backbone,
                "scheduler": self.args.scheduler,
            }
        )
        if self.args.val_dataset_dir is not None:
            self.model_meta_info["data"]["val_dataset_dir"] = self.args.val_dataset_dir

    def get_extra_norm_config(self):
        if self.args.norm_type == "limits":
            return {
                "out_min": -1.0,
                "out_max": 1.0,
            }
        return super().get_extra_norm_config()

    def set_data_stats(self):
        AddPercentileClippedWrenchToRmbData(
            self.args.dataset_dir,
            overwrite=True,
            src_key=self.args.wrench_source_key,
            dst_key=self.model_meta_info["wrench"]["key"],
        ).run()

        super().set_data_stats()

        all_image_feature = []
        all_wrench = []
        clip_min = None
        clip_max = None
        image_feature_key = self.model_meta_info["data"]["image_feature_key"]
        wrench_key = self.model_meta_info["wrench"]["key"]
        source_key = self.model_meta_info["wrench"]["source_key"]
        for filename in self.all_filenames:
            with RmbData(filename) as rmb_data:
                image_feature = convert_data_to_policy(
                    get_skipped_data_seq(
                        rmb_data[image_feature_key][:],
                        image_feature_key,
                        self.args.skip,
                    ),
                    image_feature_key,
                )
                wrench = get_skipped_data_seq(
                    rmb_data[wrench_key][:],
                    wrench_key,
                    self.args.skip,
                )
                try:
                    file_clip_min = np.asarray(
                        rmb_data.attrs[wrench_key + "_clip_min"],
                        dtype=np.float64,
                    )
                    file_clip_max = np.asarray(
                        rmb_data.attrs[wrench_key + "_clip_max"],
                        dtype=np.float64,
                    )
                    file_source_key = rmb_data.attrs[wrench_key + "_source_key"]
                except KeyError as e:
                    raise KeyError(f"{e}: {filename}") from e
                if isinstance(file_source_key, bytes):
                    file_source_key = file_source_key.decode()
                assert file_source_key == source_key, filename
                if clip_min is None:
                    clip_min = file_clip_min
                    clip_max = file_clip_max
                else:
                    assert np.allclose(clip_min, file_clip_min), filename
                    assert np.allclose(clip_max, file_clip_max), filename
            all_image_feature.append(image_feature)
            all_wrench.append(wrench)

        all_image_feature = np.concatenate(all_image_feature, dtype=np.float64)
        all_wrench = np.concatenate(all_wrench, dtype=np.float64)
        self.model_meta_info["image_feature"] = self.calc_stats_from_seq(
            all_image_feature
        )
        self.model_meta_info["wrench"].update(self.calc_stats_from_seq(all_wrench))
        self.model_meta_info["wrench"]["percentile_clip"] = {
            "key": wrench_key,
            "source_key": source_key,
            "min": clip_min,
            "max": clip_max,
        }

    def setup_dataset(self):
        if self.args.val_dataset_dir is None:
            super().setup_dataset()
            return

        if self.args.enable_rmb_cache and self.args.use_cached_dataset:
            raise ValueError(
                f"[{self.__class__.__name__}] Both 'enable_rmb_cache' and "
                "'use_cached_dataset' options cannot be True at the same time."
            )

        self.set_data_stats()
        self.add_clipped_wrench_to_validation_data()
        self.train_dataloader = self.make_dataloader(
            self.all_filenames,
            shuffle=True,
        )
        self.val_dataloader = self.make_dataloader(
            self.val_filenames,
            shuffle=False,
        )
        self.writer = SummaryWriter(self.args.checkpoint_dir)
        self.print_dataset_info()

    def add_clipped_wrench_to_validation_data(self):
        clip_info = self.model_meta_info["wrench"]["percentile_clip"]
        dst_key = clip_info["key"]
        source_key = clip_info["source_key"]
        clip_min = clip_info["min"]
        clip_max = clip_info["max"]

        print(
            f"[{self.__class__.__name__}] Add training-clipped wrench "
            f"'{dst_key}' to validation data."
        )
        for filename in self.val_filenames:
            with RmbData(filename, mode="r+") as rmb_data:
                if dst_key in rmb_data.keys():
                    del rmb_data.h5file[dst_key]
                wrench = np.asarray(rmb_data[source_key][:])
                rmb_data.h5file[dst_key] = np.clip(
                    wrench,
                    clip_min,
                    clip_max,
                ).astype(wrench.dtype, copy=False)
                rmb_data.attrs[dst_key + "_source_key"] = source_key
                rmb_data.attrs[dst_key + "_clip_min"] = clip_min
                rmb_data.attrs[dst_key + "_clip_max"] = clip_max

    def setup_policy(self):
        self.model_meta_info["policy"]["args"] = {
            "image_feature_dim": len(self.model_meta_info["image_feature"]["example"]),
            "state_dim": len(self.model_meta_info["state"]["example"]),
            "action_dim": len(self.model_meta_info["action"]["example"]),
            "wrench_dim": len(self.model_meta_info["wrench"]["example"]),
            "num_objects": len(WrenchPredictor4Dataset.OBJECT_KEY_TO_ID),
            "pb_dim": self.args.pb_dim,
            "horizon": self.args.horizon,
            "n_obs_steps": self.args.n_obs_steps,
            "hidden_dim": self.args.hidden_dim,
            "nhead": self.args.nhead,
            "num_encoder_layers": self.args.num_encoder_layers,
            "num_decoder_layers": self.args.num_decoder_layers,
            "dim_feedforward": self.args.dim_feedforward,
            "dropout": self.args.dropout,
            "output_head": self.args.output_head,
            "wrench_loss_weight": self.args.wrench_loss_weight,
        }

        self.policy = WrenchPredictor4Model(**self.model_meta_info["policy"]["args"])
        self.policy.cuda()

        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
            betas=(0.95, 0.999),
            eps=1e-8,
        )
        self.lr_scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=100,
            num_training_steps=(len(self.train_dataloader) * self.args.num_epochs),
        )

        self.print_policy_info()
        print(f"  - horizon: {self.args.horizon}, obs steps: {self.args.n_obs_steps}")
        print(
            f"  - trajectory dim: {self.policy.trajectory_dim}, image feature dim: {self.policy.image_feature_dim}, wrench dim: {self.policy.wrench_dim}"
        )

    def train_loop(self):
        for epoch in tqdm(range(self.args.num_epochs)):
            self.policy.train()
            batch_result_list = []
            for data in self.train_dataloader:
                batch = {key: value.cuda() for key, value in data.items()}
                batch_result = self.policy.compute_loss(batch)
                loss = batch_result["loss"]
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.lr_scheduler.step()
                batch_result["lr"] = self.lr_scheduler.get_last_lr()[0]
                batch_result_list.append(self.detach_batch_result(batch_result))
            self.log_epoch_summary(batch_result_list, "train", epoch)

            with torch.inference_mode():
                self.policy.eval()
                batch_result_list = []
                for data in self.val_dataloader:
                    batch = {key: value.cuda() for key, value in data.items()}
                    batch_result = self.policy.compute_loss(batch)
                    batch_result_list.append(self.detach_batch_result(batch_result))
                epoch_summary = self.log_epoch_summary(batch_result_list, "val", epoch)
                self.update_best_ckpt(epoch_summary, policy=self.policy)

            if epoch % max(self.args.num_epochs // 10, 1) == 0:
                self.save_current_ckpt(f"epoch{epoch:0>4}", policy=self.policy)

        self.save_current_ckpt("last", policy=self.policy)
        self.save_best_ckpt()

    def update_best_ckpt(self, epoch_summary, policy=None):
        if epoch_summary["loss"] < self.best_ckpt_info["loss"]:
            self.best_ckpt_info = {
                "epoch": epoch_summary["epoch"],
                "loss": epoch_summary["loss"],
                "state_dict": copy.deepcopy(policy.state_dict()),
            }
