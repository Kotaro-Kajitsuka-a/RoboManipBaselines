import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.ops.misc import FrozenBatchNorm2d


class WrenchPredictorModel(nn.Module):
    MATERIAL_PROPERTY_DIM = 9

    def __init__(
        self,
        state_dim,
        wrench_dim,
        chunk_size,
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
        self.chunk_size = chunk_size
        self.camera_names = camera_names

        # ==== (1) CNN encoder Resnet ====
        resnet = resnet18(
            weights=ResNet18_Weights.IMAGENET1K_V1,
            norm_layer=FrozenBatchNorm2d,
        )
        self.cnn = nn.Sequential(*list(resnet.children())[:-2])
        resnet_out_dim = 512
        self.input_proj_image = nn.Conv2d(
            resnet_out_dim, hidden_dim, kernel_size=1
        )

        image_height, image_width = image_shape
        with torch.no_grad():
            dummy_image = torch.zeros(1, 3, image_height, image_width)
            feature_height, feature_width = self.cnn(dummy_image).shape[-2:]
        self.image_pos_embed = nn.Parameter(
            torch.zeros(1, feature_height * feature_width, hidden_dim)
        )
        nn.init.trunc_normal_(self.image_pos_embed, std=0.02)

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

        num_image_queries = 1
        query_dim = 64
        self.image_query_embed = nn.Parameter(torch.randn(num_image_queries, query_dim))

        nheads_for_cross_attn = 4
        self.cross_attn_decoder = nn.MultiheadAttention(
            embed_dim=query_dim, num_heads=nheads_for_cross_attn, batch_first=True
        )
        self.query_proj = nn.Linear(hidden_dim, query_dim)

        state_material_feature_dim = 64
        self.state_material_encoder = nn.Sequential(
            nn.Linear(state_dim + self.MATERIAL_PROPERTY_DIM, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, state_material_feature_dim),
            nn.ReLU(inplace=True),
        )

        mlp_input_dim = query_dim * num_image_queries + state_material_feature_dim
        self.output_mlp = nn.Sequential(
            nn.LayerNorm(mlp_input_dim),
            nn.Linear(mlp_input_dim, mlp_input_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(mlp_input_dim // 2, chunk_size * self.wrench_dim),
        )

    def forward(self, state, image, material_property):
        """
        state: batch, state_dim
        image: batch, num_cam, channel, height, width
        material_property: batch, 9
        """
        batch_size = state.shape[0]

        image_tokens = []
        for cam_id, _camera_name in enumerate(self.camera_names):
            features = self.cnn(image[:, cam_id])
            features = self.input_proj_image(features)
            tokens = features.flatten(2).transpose(1, 2)
            tokens = tokens + self.image_pos_embed
            image_tokens.append(tokens)
        image_tokens = torch.cat(image_tokens, dim=1)

        tokens = self.transformer_encoder(image_tokens)
        tokens = self.query_proj(tokens)

        queries = self.image_query_embed.unsqueeze(0).repeat(batch_size, 1, 1)
        image_context, _ = self.cross_attn_decoder(queries, tokens, tokens)
        image_context = image_context.reshape(batch_size, -1)

        state_material = torch.cat([state, material_property], dim=1)
        state_material_context = self.state_material_encoder(state_material)
        context = torch.cat([image_context, state_material_context], dim=1)
        wrench_hat = self.output_mlp(context)
        wrench_hat = wrench_hat.reshape(batch_size, self.chunk_size, self.wrench_dim)
        return wrench_hat
