import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from robo_manip_baselines.misc.futureimagination.ImageVAE import (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    conv_stack,
)

CONDITION_EMBEDDING_DIM = 32
ENCODER_CHANNELS = (3, 16, 32, 64, 128, 128)
DOWNSAMPLING_FACTOR = 2 ** (len(ENCODER_CHANNELS) - 1)
FEATURE_HEIGHT = IMAGE_HEIGHT // DOWNSAMPLING_FACTOR
FEATURE_WIDTH = IMAGE_WIDTH // DOWNSAMPLING_FACTOR
FEATURE_DIM = ENCODER_CHANNELS[-1] * FEATURE_HEIGHT * FEATURE_WIDTH


class ImageJointConditionedCVAEEncoder(nn.Module):
    def __init__(self, condition_dim, latent_dim):
        super().__init__()
        self.convs = conv_stack(ENCODER_CHANNELS, nn.Conv2d)
        self.condition_embedding = nn.Sequential(
            nn.Linear(condition_dim, CONDITION_EMBEDDING_DIM),
            nn.ReLU(),
        )
        self.mu = nn.Linear(FEATURE_DIM + CONDITION_EMBEDDING_DIM, latent_dim)
        self.log_var = nn.Linear(FEATURE_DIM + CONDITION_EMBEDDING_DIM, latent_dim)

    def forward(self, images, normalized_condition):
        image_features = self.convs(images).flatten(start_dim=1)
        condition_features = self.condition_embedding(normalized_condition)
        features = torch.cat((image_features, condition_features), dim=1)
        return self.mu(features), self.log_var(features)


class ImageJointConditionedCVAEDecoder(nn.Module):
    def __init__(self, condition_dim, latent_dim):
        super().__init__()
        self.condition_embedding = nn.Sequential(
            nn.Linear(condition_dim, CONDITION_EMBEDDING_DIM),
            nn.ReLU(),
        )
        self.linear = nn.Linear(
            latent_dim + CONDITION_EMBEDDING_DIM,
            FEATURE_DIM,
        )
        self.convs = conv_stack(ENCODER_CHANNELS[::-1], nn.ConvTranspose2d)
        self.convs[-1] = nn.Sigmoid()

    def forward(self, latents, normalized_condition):
        condition_features = self.condition_embedding(normalized_condition)
        features = self.linear(torch.cat((latents, condition_features), dim=1))
        features = features.reshape(
            -1,
            ENCODER_CHANNELS[-1],
            FEATURE_HEIGHT,
            FEATURE_WIDTH,
        )
        return self.convs(features)


class ImageJointConditionedCVAE(nn.Module):
    def __init__(
        self,
        condition_keys,
        condition_mean,
        condition_std,
        latent_dim,
    ):
        super().__init__()
        condition_mean = torch.as_tensor(condition_mean, dtype=torch.float32)
        condition_std = torch.as_tensor(condition_std, dtype=torch.float32)
        assert condition_mean.ndim == 1, condition_mean.shape
        assert condition_std.shape == condition_mean.shape, (
            condition_std.shape,
            condition_mean.shape,
        )
        assert torch.all(condition_std > 0.0), condition_std

        self.condition_keys = tuple(condition_keys)
        self.condition_dim = len(condition_mean)
        self.latent_dim = latent_dim
        self.register_buffer("condition_mean", condition_mean)
        self.register_buffer("condition_std", condition_std)
        self.encoder = ImageJointConditionedCVAEEncoder(
            self.condition_dim,
            latent_dim,
        )
        self.decoder = ImageJointConditionedCVAEDecoder(
            self.condition_dim,
            latent_dim,
        )

    def normalize_condition(self, condition):
        return (condition - self.condition_mean) / self.condition_std

    def encode(self, images, condition):
        normalized_condition = self.normalize_condition(condition)
        return self.encoder(images, normalized_condition)

    def decode(self, latents, condition):
        normalized_condition = self.normalize_condition(condition)
        return self.decoder(latents, normalized_condition)

    def reconstruct(self, images, condition):
        posterior_mean, _log_variance = self.encode(images, condition)
        return self.decode(posterior_mean, condition)

    def compute_loss(self, images, condition, kl_weight, sample_posterior):
        posterior_mean, log_variance = self.encode(images, condition)
        if sample_posterior:
            std = torch.exp(0.5 * log_variance)
            latents = posterior_mean + std * torch.randn_like(std)
        else:
            latents = posterior_mean
        reconstruction = self.decode(latents, condition)
        reconstruction_loss = F.mse_loss(reconstruction, images)
        kl_loss = -0.5 * torch.mean(
            torch.sum(
                1.0 + log_variance - posterior_mean.square() - log_variance.exp(),
                dim=1,
            )
        )
        return {
            "loss": reconstruction_loss + kl_weight * kl_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }

    def save(self, checkpoint_dir):
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        config = {
            "model": self.__class__.__name__,
            "image_size": [IMAGE_WIDTH, IMAGE_HEIGHT],
            "encoder_channels": list(ENCODER_CHANNELS),
            "condition_embedding_dim": CONDITION_EMBEDDING_DIM,
            "condition_keys": list(self.condition_keys),
            "condition_mean": self.condition_mean.detach().cpu().tolist(),
            "condition_std": self.condition_std.detach().cpu().tolist(),
            "latent_dim": self.latent_dim,
        }
        with (checkpoint_dir / "config.json").open("w") as file:
            json.dump(config, file, indent=2)
        torch.save(self.state_dict(), checkpoint_dir / "model.pt")


def load_image_joint_conditioned_cvae(checkpoint_dir, device="cpu"):
    checkpoint_dir = Path(checkpoint_dir)
    with (checkpoint_dir / "config.json").open() as file:
        config = json.load(file)
    assert config["model"] == ImageJointConditionedCVAE.__name__, config["model"]
    assert config["image_size"] == [IMAGE_WIDTH, IMAGE_HEIGHT], config["image_size"]
    assert config["encoder_channels"] == list(ENCODER_CHANNELS), config[
        "encoder_channels"
    ]
    assert config["condition_embedding_dim"] == CONDITION_EMBEDDING_DIM, config[
        "condition_embedding_dim"
    ]
    model = ImageJointConditionedCVAE(
        condition_keys=config["condition_keys"],
        condition_mean=config["condition_mean"],
        condition_std=config["condition_std"],
        latent_dim=config["latent_dim"],
    )
    state_dict = torch.load(
        checkpoint_dir / "model.pt",
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state_dict)
    return model.to(device)
