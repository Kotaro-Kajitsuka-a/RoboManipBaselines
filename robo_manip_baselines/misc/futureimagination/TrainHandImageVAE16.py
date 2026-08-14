from pathlib import Path

import torch
from pythae.pipelines import TrainingPipeline
from pythae.trainers import BaseTrainerConfig

from robo_manip_baselines.common import DataKey
from robo_manip_baselines.misc.futureimagination.ImageVAE import create_image_vae
from robo_manip_baselines.misc.futureimagination.TrainImageVAE import RmbImageDataset


TRAIN_DATASET_DIR = Path("robo_manip_baselines/dataset/LiftingAB_B_only")
VALIDATION_DATASET_DIR = Path(
    "robo_manip_baselines/dataset/LiftingAB_B_only_Validation"
)
OUTPUT_DIR = Path("robo_manip_baselines/checkpoint/ImageVAE/LiftingAB_B_only_hand_16")
RGB_IMAGE_KEY = DataKey.get_rgb_image_key("hand")
LATENT_DIM = 16


def main():
    assert torch.cuda.is_available(), "Image VAE training requires a CUDA GPU."

    train_dataset = RmbImageDataset(TRAIN_DATASET_DIR, RGB_IMAGE_KEY)
    validation_dataset = RmbImageDataset(VALIDATION_DATASET_DIR, RGB_IMAGE_KEY)
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
