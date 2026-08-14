from pathlib import Path

import numpy as np
import torch
from pythae.data.datasets import DatasetOutput
from pythae.pipelines import TrainingPipeline
from pythae.trainers import BaseTrainerConfig
from torch.utils.data import Dataset
from tqdm import tqdm

from robo_manip_baselines.common import DataKey, RmbData, find_rmb_files
from robo_manip_baselines.misc.futureimagination.ImageVAE import (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    create_image_vae,
)


TRAIN_DATASET_DIR = Path("robo_manip_baselines/dataset/LiftingAB_B_only")
VALIDATION_DATASET_DIR = Path("robo_manip_baselines/dataset/LiftingABValidation")
OUTPUT_DIR = Path("robo_manip_baselines/checkpoint/ImageVAE/LiftingAB_B_only_hand_9")
RGB_IMAGE_KEY = DataKey.get_rgb_image_key("hand")
LATENT_DIM = 9


class RmbImageDataset(Dataset):
    def __init__(self, dataset_dir, rgb_image_key=RGB_IMAGE_KEY):
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


def main():
    assert torch.cuda.is_available(), "Image VAE training requires a CUDA GPU."

    train_dataset = RmbImageDataset(TRAIN_DATASET_DIR)
    validation_dataset = RmbImageDataset(VALIDATION_DATASET_DIR)
    model = create_image_vae(LATENT_DIM)
    training_config = BaseTrainerConfig(
        output_dir=str(OUTPUT_DIR),
        num_epochs=100,
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
        str(OUTPUT_DIR / "final_model"),
    )


if __name__ == "__main__":
    main()
