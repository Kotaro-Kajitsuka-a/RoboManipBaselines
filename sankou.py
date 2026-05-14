import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.ops.misc import FrozenBatchNorm2d


class SpatialAttentionEncoder(nn.Module):
    def __init__(
        self,
        shape_meta_obs: dict,
        output_dim: int = 64,
        query_dim: int = 64,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        attention_pos_as_feature: bool = False,
    ):
        super().__init__()
        self.attention_pos_as_feature = attention_pos_as_feature
        self.shape_meta_obs = shape_meta_obs
        self.rgb_keys = [k for k, v in shape_meta_obs.items() if v.get("type") == "rgb"]

        # ==== (1) CNN encoder (ResNet) ====
        weights = ResNet18_Weights.IMAGENET1K_V1
        resnet = resnet18(weights=weights, norm_layer=FrozenBatchNorm2d)
        self.cnn = nn.Sequential(*list(resnet.children())[:-2])  # remove avgpool/fc
        resnet_out_dim = 512

        # ==== (2) Learnable positional embedding ====
        _, self.image_height, self.image_width = shape_meta_obs[self.rgb_keys[0]][
            "shape"
        ]
        num_patches = (self.image_width // 32) * (self.image_height // 32)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, resnet_out_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # ==== (3) Transformer encoder ====
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=resnet_out_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # ==== (4) Learnable queries & cross-attention ====
        self.query_embed = nn.ParameterDict()
        for k in self.rgb_keys:
            num_attentions = self.shape_meta_obs[k]["num_attentions"]
            self.query_embed[k] = nn.Parameter(torch.randn(num_attentions, query_dim))
        self.cross_attn_decoder = nn.MultiheadAttention(
            embed_dim=query_dim, num_heads=nhead, batch_first=True
        )
        self.query_proj = nn.Linear(resnet_out_dim, query_dim)

        # ==== (5) State encoder ====
        state_dim = shape_meta_obs["state"]["shape"][0]
        state_feature_dim = 32
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, state_feature_dim),
            nn.ReLU(),
        )

        # ==== (6) Output head ====
        concat_dim = (2 if self.attention_pos_as_feature else query_dim) * sum(
            self.shape_meta_obs[k]["num_attentions"] for k in self.rgb_keys
        ) + state_feature_dim
        self.output_mlp = nn.Sequential(
            nn.Linear(concat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
        )

        # ==== (7) Coordinate head (for 2D supervision) ====
        self.attention_pos_head = nn.Linear(query_dim, 2)

    def forward(
        self,
        obs: dict,
        attention_pos_map_target: dict = None,
        predict_action: bool = False,
    ):
        """
        Args:
            obs["state"]: [B, state_dim]
            obs["<camera_name>_rgb_image"]: [B, 3, image_height, image_width]
            attention_pos_map_target: {camera_name: [B, num_attentions, 2]}
        Returns:
            output: [B, output_dim] (for Diffusion Policy conditioning)
            attention_pos_map: {camera_name: [B, num_attentions, 2]}
        """
        B = obs["state"].shape[0]
        attention_feature_all = []
        self.attention_feature_map = {}
        self.attention_pos_map = {}

        for k in self.rgb_keys:
            camera_name = self.shape_meta_obs[k]["camera_name"]

            image = obs[k]  # [B, 3, H, W]
            x = self.cnn(image)  # [B, resnet_out_dim, H/32, W/32]
            x = x.flatten(2).transpose(1, 2)  # [B, num_patches, resnet_out_dim]
            x = x + self.pos_embed  # add positional embedding

            x = self.transformer_encoder(x)  # [B, num_patches, resnet_out_dim]
            x_proj = self.query_proj(x)  # [B, num_patches, query_dim]

            queries = self.query_embed[k].unsqueeze(0).repeat(B, 1, 1)
            attention_feature, _ = self.cross_attn_decoder(
                queries, x_proj, x_proj
            )  # [B, num_attentions, query_dim]

            attention_pos = self.attention_pos_head(
                attention_feature
            )  # [B, num_attentions, 2]
            image_scale = torch.tensor(
                [self.image_width, self.image_height],
                device=attention_pos.device,
                dtype=attention_pos.dtype,
            )
            attention_pos = attention_pos * image_scale

            if attention_pos_map_target is not None:
                attention_pos_target = attention_pos_map_target[camera_name]
                attention_feature_target = self.calc_attention_feature(
                    attention_pos_target
                )

                if predict_action:
                    attention_pos_mask = torch.isnan(attention_pos_target)
                    attention_pos = torch.where(
                        attention_pos_mask, attention_pos, attention_pos_target
                    )
                    attention_feature_mask = torch.isnan(attention_feature_target)
                    attention_feature = torch.where(
                        attention_feature_mask,
                        attention_feature,
                        attention_feature_target,
                    )
                else:
                    num_attentions = attention_feature.shape[1]
                    mask = (
                        torch.rand(
                            B, num_attentions, 1, device=attention_feature.device
                        )
                        < 0.5
                    )

                    attention_feature = torch.where(
                        mask, attention_feature, attention_feature_target
                    )

                    if self.attention_pos_as_feature:
                        attention_pos = torch.where(
                            mask, attention_pos, attention_pos_target
                        )

            if self.attention_pos_as_feature:
                attention_feature_all.append(
                    (attention_pos / image_scale).reshape(B, -1)
                )
            else:
                attention_feature_all.append(attention_feature.reshape(B, -1))

            self.attention_feature_map[camera_name] = attention_feature
            self.attention_pos_map[camera_name] = attention_pos

        attention_feature_all = torch.cat(attention_feature_all, dim=-1)

        # ==== state encoding ====
        state_feature = self.state_encoder(obs["state"])  # [B, state_feature_dim]

        # ==== fusion and output ====
        feature_all = torch.cat([attention_feature_all, state_feature], dim=-1)
        output = self.output_mlp(feature_all)  # [B, output_dim]

        return output, self.attention_pos_map

    def calc_attention_feature(self, attention_pos):
        """
        attention_pos: [B, num_attentions, 2]
        returns: attention_feature [B, num_attentions, query_dim]
        """
        A = self.attention_pos_head.weight  # [2, query_dim]
        b = self.attention_pos_head.bias  # [2]

        image_scale = torch.tensor(
            [self.image_width, self.image_height],
            device=attention_pos.device,
            dtype=torch.float32,
        )
        attention_pos = attention_pos / image_scale

        A_pinv = torch.linalg.pinv(A, rcond=1e-5)  # [query_dim, 2]
        attention_feature = (
            attention_pos - b
        ) @ A_pinv.T  # [B, num_attentions, query_dim]

        return attention_feature
