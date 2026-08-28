import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from robo_manip_baselines.common import (
    DataKey,
    RmbData,
    convert_data_to_policy,
    find_rmb_files,
    set_random_seed,
)
from robo_manip_baselines.misc.futureimagination.ImageJointConditionedCVAE import (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    ImageJointConditionedCVAE,
)

BATCH_SIZE = 64
LEARNING_RATE = 1e-3
KL_WEIGHT = 2.5e-4
KL_WARMUP_EPOCHS = 20


def load_condition(rmb_data, condition_keys):
    condition = np.concatenate(
        [convert_data_to_policy(rmb_data[key][:], key) for key in condition_keys],
        axis=1,
    )
    return condition.astype(np.float32)


class RmbImageConditionDataset(Dataset):
    def __init__(self, dataset_dir, camera_name, condition_keys):
        rgb_image_key = DataKey.get_rgb_image_key(camera_name)
        images = []
        conditions = []
        rmb_paths = find_rmb_files(str(dataset_dir))
        assert rmb_paths, dataset_dir
        for rmb_path in tqdm(rmb_paths, desc=str(dataset_dir), unit="episode"):
            with RmbData(
                rmb_path,
                image_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
            ) as rmb_data:
                episode_images = rmb_data[rgb_image_key][:]
                episode_condition = load_condition(rmb_data, condition_keys)
            assert len(episode_images) == len(episode_condition), (
                rmb_path,
                len(episode_images),
                len(episode_condition),
            )
            images.append(episode_images)
            conditions.append(episode_condition)
        self.images = np.concatenate(images, axis=0)
        self.conditions = np.concatenate(conditions, axis=0)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = torch.from_numpy(self.images[index])
        image = image.permute(2, 0, 1).float() / 255.0
        condition = torch.from_numpy(self.conditions[index])
        return image, condition


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a joint-conditioned image CVAE on RMB data."
    )
    parser.add_argument("--train_dataset_dir", type=Path, required=True)
    parser.add_argument("--validation_dataset_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--camera_name", required=True)
    parser.add_argument("--condition_keys", nargs="+", required=True)
    parser.add_argument("--latent_dim", type=int, required=True)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def kl_weight_for_epoch(epoch):
    return KL_WEIGHT * min(epoch / KL_WARMUP_EPOCHS, 1.0)


def run_epoch(model, dataloader, optimizer, scaler, kl_weight, training):
    model.train(training)
    totals = {
        "loss": 0.0,
        "reconstruction_loss": 0.0,
        "kl_loss": 0.0,
    }
    sample_count = 0
    for images, condition in dataloader:
        images = images.cuda(non_blocking=True)
        condition = condition.cuda(non_blocking=True)
        batch_size = len(images)
        if training:
            optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                result = model.compute_loss(
                    images,
                    condition,
                    kl_weight=kl_weight,
                    sample_posterior=training,
                )
            if training:
                scaler.scale(result["loss"]).backward()
                scaler.step(optimizer)
                scaler.update()
        for key in totals:
            totals[key] += batch_size * result[key].detach().item()
        sample_count += batch_size
    return {key: value / sample_count for key, value in totals.items()}


def main():
    args = parse_args()
    assert torch.cuda.is_available(), "Image CVAE training requires a CUDA GPU."
    assert args.num_epochs > 0, args.num_epochs
    set_random_seed(args.seed)

    train_dataset = RmbImageConditionDataset(
        args.train_dataset_dir,
        args.camera_name,
        args.condition_keys,
    )
    validation_dataset = RmbImageConditionDataset(
        args.validation_dataset_dir,
        args.camera_name,
        args.condition_keys,
    )
    assert train_dataset.conditions.shape[1] == validation_dataset.conditions.shape[1]
    condition_mean = train_dataset.conditions.mean(axis=0)
    condition_std = np.clip(train_dataset.conditions.std(axis=0), 1e-6, None)
    model = ImageJointConditionedCVAE(
        condition_keys=args.condition_keys,
        condition_mean=condition_mean,
        condition_std=condition_std,
        latent_dim=args.latent_dim,
    ).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler("cuda")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
    )
    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=True,
    )

    print(f"Train frames: {len(train_dataset)}")
    print(f"Validation frames: {len(validation_dataset)}")
    print(f"Camera: {args.camera_name}")
    print(f"Condition keys: {args.condition_keys}")
    print(f"Condition dim: {model.condition_dim}")
    print(f"Latent dim: {model.latent_dim}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters()):,}")

    best_epoch = None
    best_validation_loss = float("inf")
    best_state_dict = None
    for epoch in range(1, args.num_epochs + 1):
        kl_weight = kl_weight_for_epoch(epoch)
        train_result = run_epoch(
            model,
            train_dataloader,
            optimizer,
            scaler,
            kl_weight,
            training=True,
        )
        with torch.inference_mode():
            validation_result = run_epoch(
                model,
                validation_dataloader,
                optimizer,
                scaler,
                KL_WEIGHT,
                training=False,
            )
        print(
            f"Epoch {epoch:03d}: "
            f"train={train_result['loss']:.6f}, "
            f"validation={validation_result['loss']:.6f}, "
            f"reconstruction={validation_result['reconstruction_loss']:.6f}, "
            f"KL={validation_result['kl_loss']:.6f}, "
            f"KL weight={kl_weight:.7f}"
        )
        if validation_result["loss"] < best_validation_loss:
            best_epoch = epoch
            best_validation_loss = validation_result["loss"]
            best_state_dict = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    assert best_state_dict is not None
    model.load_state_dict(best_state_dict)
    final_model_dir = args.output_dir / "final_model"
    model.save(final_model_dir)
    training_config = {
        "train_dataset_dir": str(args.train_dataset_dir),
        "validation_dataset_dir": str(args.validation_dataset_dir),
        "camera_name": args.camera_name,
        "condition_keys": args.condition_keys,
        "latent_dim": args.latent_dim,
        "num_epochs": args.num_epochs,
        "seed": args.seed,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "kl_weight": KL_WEIGHT,
        "kl_warmup_epochs": KL_WARMUP_EPOCHS,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
    }
    with (final_model_dir / "training_config.json").open("w") as file:
        json.dump(training_config, file, indent=2)
    print(f"Best validation loss: {best_validation_loss:.6f} at epoch {best_epoch}")
    print(f"Saved model: {final_model_dir}")


if __name__ == "__main__":
    main()
