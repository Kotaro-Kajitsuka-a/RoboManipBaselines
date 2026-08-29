import argparse
from pathlib import Path

import numpy as np
import torch
from pythae.data.datasets import DatasetOutput
from pythae.pipelines import TrainingPipeline
from pythae.trainers import BaseTrainerConfig
from torch.utils.data import Dataset
from tqdm import tqdm

from robo_manip_baselines.common import (
    DataKey,
    RmbData,
    find_rmb_files,
    set_random_seed,
)
from robo_manip_baselines.misc.futureimagination.ImageVAE import (
    ARCHITECTURE_BASELINE,
    ARCHITECTURES,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    create_image_vae,
)


class RmbImageDataset(Dataset):
    def __init__(self, dataset_dir, rgb_image_key):
        videos = []
        for rmb_path in tqdm(find_rmb_files(str(dataset_dir)), desc=str(dataset_dir)):
            with RmbData(rmb_path, image_size=(IMAGE_WIDTH, IMAGE_HEIGHT)) as rmb_data:
                videos.append(rmb_data[rgb_image_key][:])
        self.images = np.concatenate(videos, axis=0)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = torch.from_numpy(self.images[index])
        image = image.permute(2, 0, 1).float() / 255.0
        return DatasetOutput(data=image)


def parse_args():
    parser = argparse.ArgumentParser(description="Train an image VAE on RMB data.")
    parser.add_argument("--train_dataset_dir", type=Path, required=True)
    parser.add_argument("--validation_dataset_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--camera_name", required=True)
    parser.add_argument("--latent_dim", type=int, required=True)
    parser.add_argument(
        "--architecture",
        choices=ARCHITECTURES,
        default=ARCHITECTURE_BASELINE,
    )
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--beta", type=float, default=1.0)
    return parser.parse_args()


def main():
    args = parse_args()
    assert torch.cuda.is_available(), "Image VAE training requires a CUDA GPU."
    assert args.beta >= 0.0, args.beta
    set_random_seed(args.seed)

    rgb_image_key = DataKey.get_rgb_image_key(args.camera_name)
    train_dataset = RmbImageDataset(args.train_dataset_dir, rgb_image_key)
    validation_dataset = RmbImageDataset(args.validation_dataset_dir, rgb_image_key)
    model = create_image_vae(
        args.latent_dim,
        args.architecture,
        args.beta,
    )
    print(f"Model class: {type(model).__name__}")
    print(f"Architecture: {args.architecture}")
    print(f"Beta: {args.beta}")
    print(f"Seed: {args.seed}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters()):,}")
    training_config = BaseTrainerConfig(
        output_dir=str(args.output_dir),
        num_epochs=args.num_epochs,
        learning_rate=1e-3,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        steps_saving=10,
        seed=args.seed,
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
