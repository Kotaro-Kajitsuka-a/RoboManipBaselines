import einops
import torch
from diffusers.models import AutoencoderKL

#from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict
from torch import nn


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class BaseAutoencoder(nn.Module, ABC):
    """Abstract base class for all autoencoders used in the world model.

    Subclasses **must** implement:
      - ``encode(x)``  – maps pixel tensors to latents.
      - ``decode(z)``  – maps latents back to pixel tensors.
      - ``latent_dim`` – the channel/feature dimensionality of the latent space.
    """

    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode pixel frames to latent representations.

        Parameters
        ----------
        x : (B, T, H, W, C) float tensor in [0, 1].

        Returns
        -------
        z : (B, T, h, w, C_latent) or (B, T, N, C_latent).
        """
        ...

    @abstractmethod
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representations back to pixel space.

        Parameters
        ----------
        z : Latent tensor produced by ``encode()``.

        Returns
        -------
        x_rec : (B, T, H, W, C) float tensor approximately in [0, 1].
        """
        ...

    @property
    @abstractmethod
    def latent_dim(self) -> int:
        """Channel / feature dimensionality of the latent space."""
        ...

    @property
    def temporal_downsample_factor(self) -> int:
        """Factor by which the encoder reduces the temporal dimension.

        Default is 1 (no temporal reduction).  Encoders with 3-D tubelet
        embeddings (e.g. Qwen, V-JEPA 2) override this to 2.
        """
        return 1


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_autoencoder(config: Dict[str, Any]) -> BaseAutoencoder:
    """Instantiate an autoencoder from a configuration dict.

    Parameters
    ----------
    config : dict
        Must contain ``"encoder_type"`` (one of ``"vae"``, ``"rae"``,
        ``"scale_rae_siglip"``, ``"scale_rae_webssl"``).  Type-specific
        parameters are passed through under the corresponding key
        (e.g. ``"rae_params"``).

    Returns
    -------
    BaseAutoencoder
    """
    encoder_type = config["encoder_type"]

    if encoder_type == "vae":
        # from .encoders.vae import VAE  #it's already defined below

        return VAE()

    # if encoder_type == "rae":
    #     from .encoders.rae import RAE

    #     rae_params = config.get("rae_params", {})
    #     return RAE(**rae_params)

    # if encoder_type in ("scale_rae_siglip", "scale_rae_webssl"):
    #     from .encoders.scale_rae import ScaleRAE

    #     encoder_name = encoder_type.replace("scale_rae_", "")
    #     return ScaleRAE(
    #         encoder_name=encoder_name,
    #         decoder_config_path=config.get("scale_rae_decoder_config"),
    #         pretrained_decoder_path=config.get("pretrained_decoder_path"),
    #         normalization_stat_path=config.get("encoder_normalization_stat_path"),
    #     )

    # if encoder_type == "qwen":
    #     from .encoders.qwen import QwenEncoderWrapper
    #     return QwenEncoderWrapper(
    #         model_path=config.get("qwen_model_path", "Qwen/Qwen2.5-VL-3B-Instruct"),
    #         mode=config.get("qwen_mode", "video")
    #     )

    # if encoder_type == "vjepa2":
    #     from .encoders.vjepa2 import VJEPA2EncoderWrapper
    #     return VJEPA2EncoderWrapper(
    #         model_size=config.get("vjepa2_model_size", "vitl"),
    #         checkpoint_path=config.get("vjepa2_checkpoint_path"),
    #         input_size=config.get("vjepa2_input_size", 256),
    #     )

    # if encoder_type == "cosmos":
    #     from .encoders.cosmos import CosmosTokenizerWrapper
    #     return CosmosTokenizerWrapper(
    #         checkpoint_dir=config.get("cosmos_checkpoint_dir"),
    #     )

    # if encoder_type == "vavae":
    #     from .encoders.vavae import VAVAEWrapper
    #     return VAVAEWrapper(
    #         checkpoint_path=config.get("vavae_checkpoint_path"),
    #     )

    raise ValueError(f"Unknown encoder type: {encoder_type}")


def encoder_config_from_args(args) -> Dict[str, Any]:
    """Convert CLI *args* (from ``launch.py``) to an autoencoder config dict.

    This is the single place where argparse attributes are translated into
    the dict format expected by :func:`create_autoencoder`.
    """
    config: Dict[str, Any] = {"encoder_type": args.encoder_type}
    return config


class VAE(BaseAutoencoder):
    def __init__(self):
        super().__init__()
        self.vae = AutoencoderKL.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers", subfolder="vae"
        )
        self.vae.eval().requires_grad_(False)
        self.vae.to(torch.bfloat16)

    @property
    def latent_dim(self) -> int:
        return self.vae.config.latent_channels

    def _chunked(self, fn, x: torch.Tensor, chunk: int = 64) -> torch.Tensor:
        if x.shape[0] <= chunk:
            return fn(x)
        return torch.cat([fn(x[i:i + chunk]) for i in range(0, x.shape[0], chunk)])

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        B, T, H, W, C = x.shape
        x_in = einops.rearrange(x, "b t h w c -> (b t) c h w")
        x_in = x_in * 2 - 1

        with torch.no_grad():
            z = self._chunked(lambda x: self.vae.encode(x).latent_dist.sample(), x_in)

        z = z * self.vae.config.scaling_factor
        z = einops.rearrange(z, "(b t) c h w -> b t h w c", b=B, t=T)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        B, T, H, W, C = z.shape
        z_in = einops.rearrange(z, "b t h w c -> (b t) c h w")
        z_in = z_in / self.vae.config.scaling_factor

        with torch.no_grad():
            x = self._chunked(lambda x: self.vae.decode(x, return_dict=False)[0], z_in)

        x = (x + 1) / 2
        x = einops.rearrange(x, "(b t) c h w -> b t h w c", b=B, t=T)
        return x

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder_type", type=str, default="vae")
    args = parser.parse_args()

    config = encoder_config_from_args(args)
    autoencoder = create_autoencoder(config)
    print(f"Created autoencoder: {autoencoder}")