from pythae.models import AE, AEConfig
from pythae.models.base.base_utils import ModelOutput
from pythae.models.nn import BaseDecoder, BaseEncoder
from torch import nn


SD3_LATENT_DIM = 16 * 12 * 16
COMPACT_LATENT_DIM = 12


class SD3LatentAEEncoder(BaseEncoder):
    def __init__(self, latent_dim=COMPACT_LATENT_DIM):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(SD3_LATENT_DIM, 1024),
            nn.SiLU(),
            nn.Linear(1024, 256),
            nn.SiLU(),
            nn.Linear(256, latent_dim),
        )

    def forward(self, features):
        return ModelOutput(embedding=self.layers(features))


class SD3LatentAEDecoder(BaseDecoder):
    def __init__(self, latent_dim=COMPACT_LATENT_DIM):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 1024),
            nn.SiLU(),
            nn.Linear(1024, SD3_LATENT_DIM),
        )

    def forward(self, latents):
        return ModelOutput(reconstruction=self.layers(latents))


def create_sd3_latent_ae(latent_dim=COMPACT_LATENT_DIM):
    config = AEConfig(
        input_dim=(SD3_LATENT_DIM,),
        latent_dim=latent_dim,
    )
    return AE(
        model_config=config,
        encoder=SD3LatentAEEncoder(latent_dim),
        decoder=SD3LatentAEDecoder(latent_dim),
    )
