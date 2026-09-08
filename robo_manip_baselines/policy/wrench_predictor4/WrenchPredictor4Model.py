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
        mlp_num_hidden_layers=3,
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
        assert self.output_head in ("mlp", "mlp_only"), self.output_head

        self.material_property = nn.Embedding(num_objects, pb_dim)
        with torch.no_grad():
            for object_id in range(num_objects):
                self.material_property.weight[object_id].fill_(object_id * 0.2)
        input_dim = (
            n_obs_steps * (image_feature_dim + state_dim)
            + self.n_action_condition_steps * action_dim
            + pb_dim
        )
        if self.output_head == "mlp":
            self.image_feature_proj = nn.Linear(image_feature_dim, hidden_dim)
            self.state_proj = nn.Linear(state_dim, hidden_dim)
            self.action_proj = nn.Linear(action_dim, hidden_dim)
            self.pb_proj = nn.Linear(pb_dim, hidden_dim)
            num_condition_tokens = 2 * n_obs_steps + self.n_action_condition_steps + 1
            self.condition_pos_embed = nn.Parameter(
                torch.zeros(num_condition_tokens, hidden_dim)
            )
            input_dim = num_condition_tokens * hidden_dim
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
        if mlp_num_hidden_layers < 1:
            raise ValueError("mlp_num_hidden_layers must be at least 1")
        layers = [nn.Flatten(start_dim=1)]
        for _ in range(mlp_num_hidden_layers):
            layers.extend(
                [
                    nn.Linear(input_dim, dim_feedforward),
                    nn.LayerNorm(dim_feedforward),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            input_dim = dim_feedforward
        layers.append(nn.Linear(dim_feedforward, horizon * self.trajectory_dim))
        self.output_mlp = nn.Sequential(*layers)

        if self.output_head == "mlp":
            nn.init.normal_(self.condition_pos_embed, std=0.02)

        print("WrenchPredictor4 params: %e" % sum(p.numel() for p in self.parameters()))
        print(
            "Material PB params: %e"
            % sum(p.numel() for p in self.material_property.parameters())
        )

    @classmethod
    def from_checkpoint(cls, policy_args, state_dict):
        """Load WP4 weights, folding legacy MLP input projections into its first layer."""
        policy_args = dict(policy_args)
        # Checkpoints predating configurable depth used one hidden layer.
        policy_args.setdefault("mlp_num_hidden_layers", 1)
        model = cls(**policy_args)
        if (
            model.output_head == "mlp_only"
            and "image_feature_proj.weight" in state_dict
        ):
            state_dict = dict(state_dict)
            pos_embed = state_dict.pop("condition_pos_embed")
            weight = state_dict["output_mlp.1.weight"].reshape(-1, *pos_embed.shape)
            bias = state_dict["output_mlp.1.bias"].clone()
            input_weights = []
            offset = 0
            for name, num_steps in (
                ("image_feature", model.n_obs_steps),
                ("state", model.n_obs_steps),
                ("action", model.n_action_condition_steps),
                ("pb", 1),
            ):
                proj_weight = state_dict.pop(f"{name}_proj.weight")
                proj_bias = state_dict.pop(f"{name}_proj.bias")
                token_weight = weight[:, offset : offset + num_steps]
                input_weights.append((token_weight @ proj_weight).flatten(1))
                token_bias = proj_bias + pos_embed[offset : offset + num_steps]
                bias += (token_weight * token_bias).sum(dim=(1, 2))
                offset += num_steps
            state_dict["output_mlp.1.weight"] = torch.cat(input_weights, dim=1)
            state_dict["output_mlp.1.bias"] = bias
        model.load_state_dict(state_dict)
        return model

    def forward(self, batch, material_property=None):
        image_feature = batch["image_feature"][:, : self.n_obs_steps]
        state = batch["state"][:, : self.n_obs_steps]
        action = batch["action"][:, self.n_obs_steps - 1 : self.horizon - 1]
        if material_property is None:
            material_property = self.material_property(batch["object_id"])

        if self.output_head == "mlp":
            condition_tokens = torch.cat(
                [
                    self.image_feature_proj(image_feature),
                    self.state_proj(state),
                    self.action_proj(action),
                    self.pb_proj(material_property).unsqueeze(1),
                ],
                dim=1,
            )
            condition = self.encoder(
                condition_tokens + self.condition_pos_embed.unsqueeze(0)
            )
        else:
            condition = torch.cat(
                [
                    image_feature.flatten(1),
                    state.flatten(1),
                    action.flatten(1),
                    material_property,
                ],
                dim=1,
            )
        trajectory = self.output_mlp(condition).reshape(
            condition.shape[0],
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
