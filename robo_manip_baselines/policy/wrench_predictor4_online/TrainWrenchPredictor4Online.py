import argparse
import csv
import datetime
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from robo_manip_baselines.common import find_rmb_files, set_random_seed
from robo_manip_baselines.policy.wrench_predictor4.WrenchPredictor4Model import (
    WrenchPredictor4Model,
)

from .WrenchPredictor4OnlineDataset import WrenchPredictor4OnlineDataset


class TrainWrenchPredictor4Online:
    """Adapt one standalone PB from one episode while keeping the model frozen."""

    NUM_TRAINED_OBJECTS = 3

    def __init__(self):
        self.setup_args()
        set_random_seed(self.args.seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.setup_rmb_file()
        self.load_model_meta_info()
        self.setup_dataset()
        self.setup_policy()

    def setup_args(self):
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        parser.add_argument(
            "--dataset_dir",
            type=str,
            required=True,
            help="one RMB episode or a directory containing exactly one RMB episode",
        )
        parser.add_argument(
            "--pretrained_checkpoint",
            type=str,
            required=True,
            help="pretrained WrenchPredictor4 checkpoint",
        )
        parser.add_argument(
            "--checkpoint_dir",
            type=str,
            default=None,
            help="directory for the adapted PB and its trajectory",
        )
        parser.add_argument(
            "--lr",
            type=float,
            default=1e-2,
            help="learning rate applied only to the online PB",
        )
        parser.add_argument("--seed", type=int, default=42, help="random seed")
        self.args = parser.parse_args(sys.argv[1:])

        if self.args.checkpoint_dir is None:
            episode_name = os.path.basename(os.path.normpath(self.args.dataset_dir))
            output_name = (
                f"{episode_name}_WrenchPredictor4Online_"
                f"{datetime.datetime.now():%Y%m%d_%H%M%S}"
            )
            self.args.checkpoint_dir = os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "checkpoint",
                "WrenchPredictor4Online",
                output_name,
            )

    def setup_rmb_file(self):
        self.filenames = find_rmb_files(self.args.dataset_dir)
        assert len(self.filenames) == 1, (
            f"WrenchPredictor4Online requires exactly one RMB episode, "
            f"but found {len(self.filenames)} in {self.args.dataset_dir}"
        )

    def load_model_meta_info(self):
        checkpoint_path = os.path.abspath(self.args.pretrained_checkpoint)
        assert os.path.isfile(checkpoint_path), checkpoint_path

        meta_info_path = os.path.join(
            os.path.dirname(checkpoint_path),
            "model_meta_info.pkl",
        )
        assert os.path.isfile(meta_info_path), meta_info_path
        with open(meta_info_path, "rb") as file:
            self.model_meta_info = pickle.load(file)

        assert self.model_meta_info["policy"]["name"] == "WrenchPredictor4"
        policy_args = self.model_meta_info["policy"]["args"]
        assert policy_args["pb_dim"] == 1, policy_args["pb_dim"]
        assert policy_args["num_objects"] >= self.NUM_TRAINED_OBJECTS

    def setup_dataset(self):
        self.dataset = WrenchPredictor4OnlineDataset(
            self.filenames,
            self.model_meta_info,
            enable_rmb_cache=False,
        )
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )

    def setup_policy(self):
        policy_args = self.model_meta_info["policy"]["args"]
        self.policy = WrenchPredictor4Model(**policy_args).to(self.device)
        self.policy.load_state_dict(
            torch.load(
                self.args.pretrained_checkpoint,
                map_location=self.device,
                weights_only=True,
            )
        )
        self.policy.requires_grad_(False)
        self.policy.eval()

        self.reference_pb = (
            self.policy.material_property.weight[: self.NUM_TRAINED_OBJECTS]
            .detach()
            .clone()
        )
        self.online_pb = torch.nn.Parameter(self.reference_pb.mean(dim=0))
        self.optimizer = torch.optim.Adam([self.online_pb], lr=self.args.lr)

    def run(self):
        os.makedirs(self.args.checkpoint_dir, exist_ok=True)
        initial_pb = self.online_pb.detach().item()
        history = [
            {
                "adaptation_step": 0,
                "observed_time": np.nan,
                "pose_loss": np.nan,
                "pb": initial_pb,
            }
        ]

        for adaptation_step, batch in enumerate(tqdm(self.dataloader), start=1):
            batch = {key: value.to(self.device) for key, value in batch.items()}
            material_property = self.online_pb.unsqueeze(0)

            self.optimizer.zero_grad()
            prediction = self.policy(batch, material_property)
            start = self.policy.n_obs_steps
            # Adapt PB only from future image-feature (object-pose) error.
            # The auxiliary wrench prediction is intentionally excluded.
            pose_loss = F.mse_loss(
                prediction["image_feature"][:, start:],
                batch["image_feature"][:, start:],
            )
            pose_loss.backward()
            assert self.online_pb.grad is not None
            self.optimizer.step()

            history.append(
                {
                    "adaptation_step": adaptation_step,
                    "observed_time": batch["observed_time"].item(),
                    "pose_loss": pose_loss.detach().item(),
                    "pb": self.online_pb.detach().item(),
                }
            )

        self.save_results(history)
        print(f"device: {self.device}")
        print(f"episode: {os.path.abspath(self.filenames[0])}")
        print(f"initial PB: {initial_pb:.6f}")
        print(f"final PB: {self.online_pb.detach().item():.6f}")
        print(f"output: {os.path.abspath(self.args.checkpoint_dir)}")

    def save_results(self, history):
        csv_path = os.path.join(self.args.checkpoint_dir, "pb_adaptation.csv")
        with open(csv_path, "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=history[0].keys())
            writer.writeheader()
            writer.writerows(history)

        result_path = os.path.join(self.args.checkpoint_dir, "adapted_pb.pt")
        torch.save(
            {
                "initial_pb": history[0]["pb"],
                "adapted_pb": self.online_pb.detach().cpu(),
                "reference_pb": self.reference_pb.cpu(),
                "lr": self.args.lr,
                "pretrained_checkpoint": os.path.abspath(
                    self.args.pretrained_checkpoint
                ),
                "episode": os.path.abspath(self.filenames[0]),
            },
            result_path,
        )

        steps = np.asarray([row["adaptation_step"] for row in history])
        pb_values = np.asarray([row["pb"] for row in history])
        figure, axis = plt.subplots(figsize=(9, 5))
        axis.plot(steps, pb_values, color="black", linewidth=2, label="online PB")
        for object_id, reference_pb in enumerate(self.reference_pb[:, 0].cpu()):
            axis.axhline(
                reference_pb.item(),
                linestyle="--",
                linewidth=1.5,
                label=f"PB{object_id}={reference_pb.item():.4f}",
            )
        axis.set_xlabel("adaptation step")
        axis.set_ylabel("PB")
        axis.set_title("Online PB adaptation from one episode")
        axis.grid(alpha=0.3)
        axis.legend(loc="best")
        figure.tight_layout()
        figure.savefig(
            os.path.join(self.args.checkpoint_dir, "pb_adaptation.png"),
            dpi=160,
        )
        plt.close(figure)

    def close(self):
        pass
