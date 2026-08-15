import torch

from robo_manip_baselines.common import DataKey, TrainBase
from robo_manip_baselines.misc.AddPercentileClippedWrenchToRmbData import (
    get_percentile_clip_wrench_key,
)
from robo_manip_baselines.policy.wrench_predictor4.TrainWrenchPredictor4 import (
    TrainWrenchPredictor4,
    get_cosine_schedule_with_warmup,
)

from .WrenchPredictor5Dataset import WrenchPredictor5Dataset
from .WrenchPredictor5Model import WrenchPredictor5Model


class TrainWrenchPredictor5(TrainWrenchPredictor4):
    DatasetClass = WrenchPredictor5Dataset
    LATENT_SHAPE = (16, 15, 20)

    def setup_args(self):
        TrainBase.setup_args(self)

    def set_additional_args(self, parser):
        parser.set_defaults(
            enable_rmb_cache=True,
            norm_type="limits",
            batch_size=8,
            num_epochs=200,
            lr=1e-4,
        )
        parser.add_argument(
            "--weight_decay", type=float, default=1e-6, help="weight decay"
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
            default=512,
            help="Transformer token dimension",
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
            default=4,
            help="number of Transformer decoder layers",
        )
        parser.add_argument(
            "--dim_feedforward",
            type=int,
            default=2048,
            help="Transformer feedforward dimension",
        )
        parser.add_argument(
            "--dropout",
            type=float,
            default=0.1,
            help="Transformer dropout",
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

    def setup_model_meta_info(self):
        TrainBase.setup_model_meta_info(self)
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
            "object_key_to_id": WrenchPredictor5Dataset.OBJECT_KEY_TO_ID,
        }
        if self.args.val_dataset_dir is not None:
            self.model_meta_info["data"]["val_dataset_dir"] = self.args.val_dataset_dir

    def setup_policy(self):
        image_feature_dim = len(self.model_meta_info["image_feature"]["example"])
        assert image_feature_dim == 16 * 15 * 20, image_feature_dim
        self.model_meta_info["policy"]["args"] = {
            "image_feature_dim": image_feature_dim,
            "state_dim": len(self.model_meta_info["state"]["example"]),
            "action_dim": len(self.model_meta_info["action"]["example"]),
            "wrench_dim": len(self.model_meta_info["wrench"]["example"]),
            "num_objects": len(WrenchPredictor5Dataset.OBJECT_KEY_TO_ID),
            "pb_dim": self.args.pb_dim,
            "horizon": self.args.horizon,
            "n_obs_steps": self.args.n_obs_steps,
            "latent_shape": self.LATENT_SHAPE,
            "hidden_dim": self.args.hidden_dim,
            "nhead": self.args.nhead,
            "num_encoder_layers": self.args.num_encoder_layers,
            "num_decoder_layers": self.args.num_decoder_layers,
            "dim_feedforward": self.args.dim_feedforward,
            "dropout": self.args.dropout,
            "wrench_loss_weight": self.args.wrench_loss_weight,
        }

        self.policy = WrenchPredictor5Model(
            **self.model_meta_info["policy"]["args"]
        ).cuda()
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
            num_training_steps=len(self.train_dataloader) * self.args.num_epochs,
        )

        self.print_policy_info()
        print(
            f"  - horizon: {self.args.horizon}, obs steps: {self.args.n_obs_steps}, "
            f"future steps: {self.policy.num_future_steps}"
        )
        print(
            f"  - latent shape: {self.LATENT_SHAPE}, "
            f"image tokens per observation: {self.policy.latent_channels}, "
            f"token dim: {self.policy.hidden_dim}"
        )
