import torch
import torch.nn as nn
import torch.nn.functional as F


class WrenchPredictor4Model(nn.Module):
    """Regressor for future wrench and absolute image feature."""

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
        dim_feedforward=1024,
        dropout=0.1,
        output_head="mlp_only",
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
        self.output_head = output_head
        self.wrench_loss_weight = wrench_loss_weight
        self.n_action_condition_steps = horizon - n_obs_steps
        assert self.n_action_condition_steps > 0, (horizon, n_obs_steps)
        assert self.wrench_loss_weight >= 0.0, self.wrench_loss_weight
        assert self.output_head in ("mlp", "mlp_only"), self.output_head

        self.material_property = nn.Embedding(num_objects, pb_dim)
        self.image_feature_proj = nn.Linear(image_feature_dim, hidden_dim)
        self.state_proj = nn.Linear(state_dim, hidden_dim)
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        self.pb_proj = nn.Linear(pb_dim, hidden_dim)

        num_condition_tokens = 2 * n_obs_steps + self.n_action_condition_steps + 1
        self.condition_pos_embed = nn.Parameter(
            torch.zeros(num_condition_tokens, hidden_dim)
        )

        if self.output_head == "mlp":
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
        self.output_mlp = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(num_condition_tokens * hidden_dim, dim_feedforward),
            nn.LayerNorm(dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, horizon * self.trajectory_dim),
        )

        nn.init.normal_(self.condition_pos_embed, std=0.02)

        print("WrenchPredictor4 params: %e" % sum(p.numel() for p in self.parameters()))
        print(
            "Material PB params: %e"
            % sum(p.numel() for p in self.material_property.parameters())
        )

    @classmethod
    def from_policy_args(cls, policy_args):
        policy_args = dict(policy_args)
        policy_args.pop("num_decoder_layers", None)
        return cls(**policy_args)

    def get_condition_tokens(self, batch, material_property=None):
        image_feature = batch["image_feature"][:, : self.n_obs_steps]
        state = batch["state"][:, : self.n_obs_steps]
        action = batch["action"][:, self.n_obs_steps - 1 : self.horizon - 1]
        if material_property is None:
            material_property = self.material_property(batch["object_id"])

        image_feature_tokens = self.image_feature_proj(image_feature)
        state_tokens = self.state_proj(state)
        action_tokens = self.action_proj(action)
        pb_token = self.pb_proj(material_property).unsqueeze(1)
        tokens = torch.cat(
            [image_feature_tokens, state_tokens, action_tokens, pb_token],
            dim=1,
        )
        return tokens + self.condition_pos_embed.unsqueeze(0)

    def forward(self, batch, material_property=None):
        condition_tokens = self.get_condition_tokens(batch, material_property)
        if self.output_head == "mlp":
            condition_tokens = self.encoder(condition_tokens)
        trajectory = self.output_mlp(condition_tokens).reshape(
            condition_tokens.shape[0],
            self.horizon,
            self.trajectory_dim,
        )
        wrench = trajectory[..., : self.wrench_dim]
        image_feature = trajectory[..., self.wrench_dim :]
        return {
            "wrench": wrench,
            "image_feature": image_feature,
        }

    def compute_loss(self, batch, material_property=None):
        pred = self.forward(batch, material_property)
        start = self.n_obs_steps
        wrench_loss = F.mse_loss(pred["wrench"][:, start:], batch["wrench"][:, start:])
        image_feature_loss = F.mse_loss(
            pred["image_feature"][:, start:],
            batch["image_feature"][:, start:],
        )
        return {
            "loss": self.wrench_loss_weight * wrench_loss + image_feature_loss,
            "wrench_loss": wrench_loss,
            "image_feature_loss": image_feature_loss,
        }

    @torch.no_grad()
    def predict(self, batch):
        return self.forward(batch)
