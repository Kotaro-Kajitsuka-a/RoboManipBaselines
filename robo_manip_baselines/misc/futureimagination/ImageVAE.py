from pythae.models import VAE, VAEConfig
from pythae.models.base.base_utils import ModelOutput
from pythae.models.nn import BaseDecoder, BaseEncoder
from torch import nn

IMAGE_HEIGHT = 96
IMAGE_WIDTH = 128
LATENT_DIM = 6
ARCHITECTURE_BASELINE = "baseline"
ARCHITECTURE_LARGE = "large"
ARCHITECTURES = (ARCHITECTURE_BASELINE, ARCHITECTURE_LARGE)


def conv_stack(channels, layer_class):
    modules = []
    for in_channels, out_channels in zip(channels, channels[1:]):
        modules.append(
            layer_class(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            )
        )
        modules.append(nn.ReLU())
    return nn.Sequential(*modules)


class BaselineImageVAEEncoder(BaseEncoder):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.convs = conv_stack((3, 16, 32, 64, 128, 128), nn.Conv2d)
        self.mu = nn.Linear(128 * 3 * 4, latent_dim)
        self.log_var = nn.Linear(128 * 3 * 4, latent_dim)

    def forward(self, images):
        features = self.convs(images).flatten(start_dim=1)
        return ModelOutput(
            embedding=self.mu(features),
            log_covariance=self.log_var(features),
        )


class BaselineImageVAEDecoder(BaseDecoder):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.linear = nn.Linear(latent_dim, 128 * 3 * 4)
        self.convs = conv_stack((128, 128, 64, 32, 16, 3), nn.ConvTranspose2d)
        self.convs[-1] = nn.Sigmoid()

    def forward(self, latents):
        features = self.linear(latents).reshape(-1, 128, 3, 4)
        return ModelOutput(reconstruction=self.convs(features))


class LargeImageVAEEncoder(BaseEncoder):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.convs = conv_stack((3, 96, 192, 384), nn.Conv2d)
        self.mu = nn.Linear(384 * 12 * 16, latent_dim)
        self.log_var = nn.Linear(384 * 12 * 16, latent_dim)

    def forward(self, images):
        features = self.convs(images).flatten(start_dim=1)
        return ModelOutput(
            embedding=self.mu(features),
            log_covariance=self.log_var(features),
        )


class LargeImageVAEDecoder(BaseDecoder):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.linear = nn.Linear(latent_dim, 384 * 12 * 16)
        self.convs = conv_stack((384, 192, 96, 3), nn.ConvTranspose2d)
        self.convs[-1] = nn.Sigmoid()

    def forward(self, latents):
        features = self.linear(latents).reshape(-1, 384, 12, 16)
        return ModelOutput(reconstruction=self.convs(features))


def create_image_vae(
    latent_dim=LATENT_DIM,
    architecture=ARCHITECTURE_BASELINE,
):
    config = VAEConfig(
        input_dim=(3, IMAGE_HEIGHT, IMAGE_WIDTH),
        latent_dim=latent_dim,
        reconstruction_loss="mse",
    )
    if architecture == ARCHITECTURE_BASELINE:
        encoder = BaselineImageVAEEncoder(latent_dim)
        decoder = BaselineImageVAEDecoder(latent_dim)
    elif architecture == ARCHITECTURE_LARGE:
        encoder = LargeImageVAEEncoder(latent_dim)
        decoder = LargeImageVAEDecoder(latent_dim)
    else:
        raise ValueError(f"Unknown ImageVAE architecture: {architecture}")
    return VAE(model_config=config, encoder=encoder, decoder=decoder)
