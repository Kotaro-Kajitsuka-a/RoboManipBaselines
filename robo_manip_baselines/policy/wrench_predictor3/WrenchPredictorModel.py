import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.ops.misc import FrozenBatchNorm2d


class WrenchPredictorModel(nn.Module):
    MATERIAL_PROPERTY_DIM = 9
    IMAGE_HISTORY_SIZE = 2

    def __init__(
        self,
        state_dim,
        wrench_dim,
        horizon,
        camera_names,
        image_shape,
        hidden_dim,
        dim_feedforward,
        enc_layers,
        nheads,
        dropout=0.1,
        pre_norm=False,
    ):
        super().__init__()
        self.wrench_dim = wrench_dim
        self.horizon = horizon
        self.camera_names = camera_names

        resnet = resnet18(
            weights=ResNet18_Weights.IMAGENET1K_V1,
            norm_layer=FrozenBatchNorm2d,
        )
        self.cnn = nn.Sequential(*list(resnet.children())[:-2])
        resnet_out_dim = 512
        self.input_proj_image = nn.Conv2d(resnet_out_dim, hidden_dim, kernel_size=1)
        self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)
        self.input_proj_material_property = nn.Linear(
            self.MATERIAL_PROPERTY_DIM, hidden_dim
        )
        image_height, image_width = image_shape
        with torch.no_grad():
            dummy_image = torch.zeros(1, 3, image_height, image_width)
            feature_height, feature_width = self.cnn(dummy_image).shape[-2:]
        self.spatial_pos_embed = nn.Parameter(
            torch.zeros(1, 1, 1, feature_height * feature_width, hidden_dim)
        )
        self.camera_pos_embed = nn.Parameter(
            torch.zeros(1, 1, len(camera_names), 1, hidden_dim)
        )
        self.time_pos_embed = nn.Parameter(
            torch.zeros(1, self.IMAGE_HISTORY_SIZE, 1, 1, hidden_dim)
        )
        nn.init.trunc_normal_(self.spatial_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.camera_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.time_pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=pre_norm,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=enc_layers,
            norm=nn.LayerNorm(hidden_dim),
        )
        self.output_mlp = nn.Sequential(
            nn.LayerNorm(3 * hidden_dim),
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, horizon * self.wrench_dim),
        )

    def forward(self, state, image, material_property):
        """
        state: batch, state_dim
        image: batch, num_cam, history, channel, height, width
        material_property: batch, 9
        """
        batch_size = state.shape[0]
        assert image.ndim == 6, image.shape
        assert image.shape[1] == len(self.camera_names), image.shape
        assert image.shape[2] == self.IMAGE_HISTORY_SIZE, image.shape

        camera_num = image.shape[1]
        image = image.moveaxis(1, 2)
        image = image.reshape(
            batch_size * self.IMAGE_HISTORY_SIZE * camera_num,
            *image.shape[3:],
        )
        features = self.cnn(image)
        features = self.input_proj_image(features)
        tokens = features.flatten(2).transpose(1, 2)
        token_num = tokens.shape[1]
        tokens = tokens.reshape(
            batch_size,
            self.IMAGE_HISTORY_SIZE,
            camera_num,
            token_num,
            -1,
        )
        tokens = (
            tokens
            + self.spatial_pos_embed
            + self.camera_pos_embed
            + self.time_pos_embed
        )
        image_tokens = tokens.reshape(batch_size, -1, tokens.shape[-1])

        robot_state_token = self.input_proj_robot_state(state)
        prefix_tokens = robot_state_token.unsqueeze(1)

        tokens = torch.cat([prefix_tokens, image_tokens], dim=1)
        tokens = self.transformer_encoder(tokens)

        robot_state_context = tokens[:, 0]
        image_context = tokens[:, 1:].mean(dim=1)
        material_property_context = self.input_proj_material_property(material_property)
        context = torch.cat(
            [robot_state_context, image_context, material_property_context], dim=1
        )
        wrench_hat = self.output_mlp(context)
        wrench_hat = wrench_hat.reshape(batch_size, self.horizon, self.wrench_dim)
        return wrench_hat
