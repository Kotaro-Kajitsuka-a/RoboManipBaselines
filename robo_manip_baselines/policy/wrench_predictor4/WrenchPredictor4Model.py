import torch
import torch.nn as nn
import torch.nn.functional as F


class WrenchPredictor4Model(nn.Module):
    """Transformer regressor for future wrench and marker/image feature."""

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
        hidden_dim=256,
        nhead=8,
        num_encoder_layers=4,
        num_decoder_layers=2,
        dim_feedforward=1024,
        dropout=0.1,
        image_feature_target_mode="absolute",
        output_head="decoder",
        wrench_loss_weight=1.0,
    ):
        super().__init__()

        self.image_feature_dim = image_feature_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.wrench_dim = wrench_dim
        self.trajectory_dim = wrench_dim + image_feature_dim
        self.pb_dim = pb_dim
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.image_feature_target_mode = image_feature_target_mode
        self.output_head = output_head
        self.wrench_loss_weight = wrench_loss_weight
        self.n_action_condition_steps = horizon - n_obs_steps
        assert self.n_action_condition_steps > 0, (horizon, n_obs_steps)
        assert self.wrench_loss_weight >= 0.0, self.wrench_loss_weight
        assert self.image_feature_target_mode in (
            "absolute",
            "delta_from_last_obs",
        ), self.image_feature_target_mode
        assert self.output_head in ("decoder", "mlp", "mlp_only"), self.output_head

        self.material_property = nn.Embedding(num_objects, pb_dim)
        self.obs_proj = nn.Linear(image_feature_dim + state_dim, hidden_dim)
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        self.pb_proj = nn.Linear(pb_dim, hidden_dim)

        num_condition_tokens = n_obs_steps + self.n_action_condition_steps + 1
        self.condition_pos_embed = nn.Parameter(
            torch.zeros(num_condition_tokens, hidden_dim)
        )

        if self.output_head != "mlp_only":
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
            )
        if self.output_head == "decoder":
            self.query_embed = nn.Parameter(torch.zeros(horizon, hidden_dim))
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
            )
            self.output_proj = nn.Linear(hidden_dim, self.trajectory_dim)
        else:
            self.output_mlp = nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.Linear(num_condition_tokens * hidden_dim, dim_feedforward),
                nn.LayerNorm(dim_feedforward),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward, horizon * self.trajectory_dim),
            )

        nn.init.normal_(self.condition_pos_embed, std=0.02)
        if self.output_head == "decoder":
            nn.init.normal_(self.query_embed, std=0.02)

        print("Transformer params: %e" % sum(p.numel() for p in self.parameters()))
        print(
            "Material PB params: %e"
            % sum(p.numel() for p in self.material_property.parameters())
        )

    def get_condition_tokens(self, batch):
        image_feature = batch["image_feature"][:, : self.n_obs_steps]
        state = batch["state"][:, : self.n_obs_steps]
        action = batch["action"][:, self.n_obs_steps - 1 : self.horizon - 1]
        material_property = self.material_property(batch["object_id"])

        obs_tokens = self.obs_proj(torch.cat([image_feature, state], dim=-1))
        action_tokens = self.action_proj(action)
        pb_token = self.pb_proj(material_property).unsqueeze(1)
        tokens = torch.cat([obs_tokens, action_tokens, pb_token], dim=1)
        return tokens + self.condition_pos_embed.unsqueeze(0)

    def forward(self, batch):
        condition_tokens = self.get_condition_tokens(batch)
        if self.output_head == "decoder":
            memory = self.encoder(condition_tokens)
            query = self.query_embed.unsqueeze(0).expand(
                condition_tokens.shape[0], -1, -1
            )
            decoded = self.decoder(query, memory)
            trajectory = self.output_proj(decoded)
        else:
            if self.output_head == "mlp":
                condition_tokens = self.encoder(condition_tokens)
            trajectory = self.output_mlp(condition_tokens).reshape(
                condition_tokens.shape[0],
                self.horizon,
                self.trajectory_dim,
            )
        wrench = trajectory[..., : self.wrench_dim]
        image_feature_target = trajectory[..., self.wrench_dim :]
        image_feature = self.restore_image_feature(batch, image_feature_target)
        return {
            "wrench": wrench,
            "image_feature": image_feature,
            "image_feature_target": image_feature_target,
        }

    def get_image_feature_ref(self, batch):
        return batch["image_feature"][:, self.n_obs_steps - 1 : self.n_obs_steps]

    def get_image_feature_target(self, batch):
        if self.image_feature_target_mode == "absolute":
            return batch["image_feature"]
        return batch["image_feature"] - self.get_image_feature_ref(batch)

    def restore_image_feature(self, batch, image_feature_target):
        if self.image_feature_target_mode == "absolute":
            return image_feature_target
        return self.get_image_feature_ref(batch) + image_feature_target

    def compute_loss(self, batch):
        pred = self.forward(batch)
        start = self.n_obs_steps
        wrench_loss = F.mse_loss(pred["wrench"][:, start:], batch["wrench"][:, start:])
        image_feature_target = self.get_image_feature_target(batch)
        image_feature_loss = F.mse_loss(
            pred["image_feature_target"][:, start:],
            image_feature_target[:, start:],
        )
        return {
            "loss": self.wrench_loss_weight * wrench_loss + image_feature_loss,
            "wrench_loss": wrench_loss,
            "image_feature_loss": image_feature_loss,
        }

    @torch.no_grad()
    def predict(self, batch):
        return self.forward(batch)
