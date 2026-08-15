import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class WrenchPredictor5Model(nn.Module):
    """Predict future SD3 VAE latents and wrench with an encoder-decoder Transformer."""

    IMAGE_TOKEN = 0
    STATE_TOKEN = 1
    ACTION_TOKEN = 2
    PB_TOKEN = 3
    LATENT_QUERY_TOKEN = 4
    WRENCH_QUERY_TOKEN = 5
    NUM_TOKEN_TYPES = 6

    def __init__(
        self,
        image_feature_dim,
        state_dim,
        action_dim,
        wrench_dim,
        num_objects,
        pb_dim,
        horizon,
        n_obs_steps,
        latent_shape,
        hidden_dim=512,
        nhead=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=2048,
        dropout=0.1,
        wrench_loss_weight=1.0,
    ):
        super().__init__()

        self.image_feature_dim = image_feature_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.wrench_dim = wrench_dim
        self.pb_dim = pb_dim
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.num_future_steps = horizon - n_obs_steps
        self.latent_shape = tuple(latent_shape)
        (
            self.latent_channels,
            self.latent_height,
            self.latent_width,
        ) = self.latent_shape
        self.latent_spatial_dim = self.latent_height * self.latent_width
        self.hidden_dim = hidden_dim
        self.wrench_loss_weight = wrench_loss_weight

        assert image_feature_dim == math.prod(self.latent_shape), (
            image_feature_dim,
            self.latent_shape,
        )
        assert 0 < n_obs_steps < horizon, (n_obs_steps, horizon)
        assert hidden_dim % nhead == 0, (hidden_dim, nhead)

        self.material_property = nn.Embedding(num_objects, pb_dim)
        self.image_feature_proj = nn.Linear(self.latent_spatial_dim, hidden_dim)
        self.state_proj = nn.Linear(state_dim, hidden_dim)
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        self.pb_proj = nn.Linear(pb_dim, hidden_dim)

        self.observation_time_embed = nn.Embedding(n_obs_steps, hidden_dim)
        self.future_time_embed = nn.Embedding(self.num_future_steps, hidden_dim)
        self.latent_channel_embed = nn.Embedding(self.latent_channels, hidden_dim)
        self.token_type_embed = nn.Embedding(self.NUM_TOKEN_TYPES, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
            norm=nn.LayerNorm(hidden_dim),
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers,
            norm=nn.LayerNorm(hidden_dim),
        )

        self.latent_output_proj = nn.Linear(hidden_dim, self.latent_spatial_dim)
        self.wrench_output_proj = nn.Linear(hidden_dim, wrench_dim)

        nn.init.normal_(self.observation_time_embed.weight, std=0.02)
        nn.init.normal_(self.future_time_embed.weight, std=0.02)
        nn.init.normal_(self.latent_channel_embed.weight, std=0.02)
        nn.init.normal_(self.token_type_embed.weight, std=0.02)

        print(
            "WrenchPredictor5 params: %e"
            % sum(parameter.numel() for parameter in self.parameters())
        )
        print(
            "Material PB params: %e"
            % sum(
                parameter.numel() for parameter in self.material_property.parameters()
            )
        )

    def get_encoder_memory(self, batch, material_property=None):
        batch_size = batch["image_feature"].shape[0]
        image_feature = batch["image_feature"][:, : self.n_obs_steps].reshape(
            batch_size,
            self.n_obs_steps,
            self.latent_channels,
            self.latent_spatial_dim,
        )
        image_tokens = (
            self.image_feature_proj(image_feature)
            + self.observation_time_embed.weight[None, :, None, :]
            + self.latent_channel_embed.weight[None, None, :, :]
            + self.token_type_embed.weight[self.IMAGE_TOKEN]
        ).reshape(
            batch_size,
            self.n_obs_steps * self.latent_channels,
            self.hidden_dim,
        )

        state = batch["state"][:, : self.n_obs_steps]
        state_tokens = (
            self.state_proj(state)
            + self.observation_time_embed.weight[None, :, :]
            + self.token_type_embed.weight[self.STATE_TOKEN]
        )

        action = batch["action"][:, self.n_obs_steps - 1 : self.horizon - 1]
        assert action.shape[1] == self.num_future_steps, action.shape
        action_tokens = (
            self.action_proj(action)
            + self.future_time_embed.weight[None, :, :]
            + self.token_type_embed.weight[self.ACTION_TOKEN]
        )

        if material_property is None:
            material_property = self.material_property(batch["object_id"])
        pb_token = (
            self.pb_proj(material_property)
            + self.token_type_embed.weight[self.PB_TOKEN]
        ).unsqueeze(1)

        encoder_tokens = torch.cat(
            [image_tokens, state_tokens, action_tokens, pb_token],
            dim=1,
        )
        return self.encoder(encoder_tokens)

    def get_decoder_queries(self, batch_size):
        latent_queries = (
            self.future_time_embed.weight[:, None, :]
            + self.latent_channel_embed.weight[None, :, :]
            + self.token_type_embed.weight[self.LATENT_QUERY_TOKEN]
        )
        wrench_queries = (
            self.future_time_embed.weight
            + self.token_type_embed.weight[self.WRENCH_QUERY_TOKEN]
        ).unsqueeze(1)
        decoder_queries = torch.cat(
            [latent_queries, wrench_queries],
            dim=1,
        ).reshape(
            1,
            self.num_future_steps * (self.latent_channels + 1),
            self.hidden_dim,
        )
        return decoder_queries.expand(batch_size, -1, -1)

    def forward(self, batch, material_property=None):
        batch_size = batch["image_feature"].shape[0]
        memory = self.get_encoder_memory(batch, material_property)
        decoder_queries = self.get_decoder_queries(batch_size)
        decoded = self.decoder(decoder_queries, memory).reshape(
            batch_size,
            self.num_future_steps,
            self.latent_channels + 1,
            self.hidden_dim,
        )

        latent_tokens = decoded[:, :, : self.latent_channels]
        image_feature = self.latent_output_proj(latent_tokens).reshape(
            batch_size,
            self.num_future_steps,
            self.image_feature_dim,
        )
        wrench = self.wrench_output_proj(decoded[:, :, self.latent_channels])
        return {
            "image_feature": image_feature,
            "wrench": wrench,
        }

    def compute_loss(self, batch, material_property=None):
        pred = self.forward(batch, material_property)
        target_image_feature = batch["image_feature"][:, self.n_obs_steps :]
        target_wrench = batch["wrench"][:, self.n_obs_steps :]
        assert pred["image_feature"].shape == target_image_feature.shape
        assert pred["wrench"].shape == target_wrench.shape

        image_feature_loss = F.mse_loss(
            pred["image_feature"],
            target_image_feature,
        )
        wrench_loss = F.mse_loss(pred["wrench"], target_wrench)
        return {
            "loss": image_feature_loss + self.wrench_loss_weight * wrench_loss,
            "image_feature_loss": image_feature_loss,
            "wrench_loss": wrench_loss,
        }

    @torch.no_grad()
    def predict(self, batch):
        return self.forward(batch)
