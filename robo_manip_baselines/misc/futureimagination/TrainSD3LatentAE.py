import argparse
from pathlib import Path

import numpy as np
import torch
from pythae.data.datasets import DatasetOutput
from pythae.pipelines import TrainingPipeline
from pythae.trainers import BaseTrainerConfig
from torch.utils.data import Dataset
from tqdm import tqdm

from robo_manip_baselines.common import RmbData, find_rmb_files
from robo_manip_baselines.misc.futureimagination.SD3LatentAE import (
    SD3_LATENT_DIM,
    create_sd3_latent_ae,
)


DEFAULT_IMAGE_FEATURE_KEY = "sd3_vae_hand"


class RmbSD3LatentDataset(Dataset):
    def __init__(self, dataset_dir, image_feature_key):
        features = []
        rmb_paths = find_rmb_files(str(dataset_dir))
        assert len(rmb_paths) > 0, dataset_dir

        for rmb_path in tqdm(rmb_paths, desc=str(dataset_dir), unit="episode"):
            with RmbData(rmb_path) as rmb_data:
                assert image_feature_key in rmb_data, (rmb_path, image_feature_key)
                episode_features = rmb_data[image_feature_key][:]
            assert episode_features.ndim == 2, episode_features.shape
            assert episode_features.shape[1] == SD3_LATENT_DIM, (
                rmb_path,
                episode_features.shape,
            )
            features.append(episode_features.astype(np.float32))

        self.features = np.concatenate(features, axis=0)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return DatasetOutput(data=torch.from_numpy(self.features[index]))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a compact AE to reconstruct frozen SD3 VAE latents."
    )
    parser.add_argument("--train_dataset_dir", type=Path, required=True)
    parser.add_argument("--validation_dataset_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--image_feature_key",
        default=DEFAULT_IMAGE_FEATURE_KEY,
    )
    parser.add_argument("--latent_dim", type=int, default=12)
    parser.add_argument("--num_epochs", type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()
    assert torch.cuda.is_available(), "SD3 latent AE training requires a CUDA GPU."

    train_dataset = RmbSD3LatentDataset(
        args.train_dataset_dir,
        args.image_feature_key,
    )
    validation_dataset = RmbSD3LatentDataset(
        args.validation_dataset_dir,
        args.image_feature_key,
    )
    print(f"Train frames: {len(train_dataset)}")
    print(f"Validation frames: {len(validation_dataset)}")
    print(f"Input feature: {args.image_feature_key} ({SD3_LATENT_DIM}D)")
    print(f"Compact latent: {args.latent_dim}D")

    model = create_sd3_latent_ae(args.latent_dim)
    training_config = BaseTrainerConfig(
        output_dir=str(args.output_dir),
        num_epochs=args.num_epochs,
        learning_rate=1e-3,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        steps_saving=10,
        seed=42,
        amp=True,
    )
    pipeline = TrainingPipeline(model, training_config)
    pipeline(train_dataset, validation_dataset)
    pipeline.trainer.save_model(
        pipeline.trainer._best_model,
        str(args.output_dir / "final_model"),
    )


if __name__ == "__main__":
    main()
