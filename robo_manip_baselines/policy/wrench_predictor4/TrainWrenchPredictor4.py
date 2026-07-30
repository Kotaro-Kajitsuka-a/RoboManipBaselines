import copy

import torch
from tqdm import tqdm

from robo_manip_baselines.policy.diffusion_world_model.TrainDiffusionWorldModel import (
    TrainDiffusionWorldModel,
)

from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.common.lr_scheduler import get_scheduler

from .WrenchPredictor4Dataset import WrenchPredictor4Dataset
from .WrenchPredictor4Model import WrenchPredictor4Model


class TrainWrenchPredictor4(TrainDiffusionWorldModel):
    DatasetClass = WrenchPredictor4Dataset

    def setup_args(self):
        super().setup_args()
        if self.args.backbone != "transformer":
            raise ValueError(
                f"[{self.__class__.__name__}] WrenchPredictor4 uses only transformer backbone."
            )

    def set_additional_args(self, parser):
        super().set_additional_args(parser)
        parser.set_defaults(backbone="transformer")
        parser.set_defaults(scheduler="ddpm")
        parser.set_defaults(horizon=16)
        parser.set_defaults(num_epochs=200)
        parser.set_defaults(lr=1e-4)
        parser.set_defaults(use_ema=False)
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
            "image_feature_target_mode": self.args.image_feature_target_mode,
            "hidden_dim": self.args.hidden_dim,
            "nhead": self.args.nhead,
            "num_encoder_layers": self.args.num_encoder_layers,
            "num_decoder_layers": self.args.num_decoder_layers,
            "dim_feedforward": self.args.dim_feedforward,
            "dropout": self.args.dropout,
            "output_head": self.args.output_head,
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
        self.lr_scheduler = get_scheduler(
            name="cosine",
            optimizer=self.optimizer,
            num_warmup_steps=100,
            num_training_steps=(len(self.train_dataloader) * self.args.num_epochs),
        )
        optimizer_to(self.optimizer, "cuda")

        self.print_policy_info()
        print(
            f"  - horizon: {self.args.horizon}, obs steps: {self.args.n_obs_steps}"
        )
        print(
            f"  - trajectory dim: {self.policy.trajectory_dim}, image feature dim: {self.policy.image_feature_dim}, wrench dim: {self.policy.wrench_dim}"
        )

    def train_loop(self):
        for epoch in tqdm(range(self.args.num_epochs)):
            self.policy.train()
            batch_result_list = []
            for data in self.train_dataloader:
                batch_result = self.policy.compute_loss(
                    dict_apply(data, lambda x: x.cuda())
                )
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
                    batch_result = self.policy.compute_loss(
                        dict_apply(data, lambda x: x.cuda())
                    )
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
