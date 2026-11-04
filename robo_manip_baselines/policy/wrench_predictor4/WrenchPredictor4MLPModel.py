import torch
import torch.nn as nn
import torch.nn.functional as F


class WrenchPredictor4MLPModel(nn.Module):
    """Three-hidden-layer MLP over concatenated observations, actions, and PB."""

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
        dim_feedforward=1024,
        dropout=0.1,
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
        self.wrench_loss_weight = wrench_loss_weight
        self.n_action_condition_steps = horizon - n_obs_steps

        self.material_property = nn.Embedding(num_objects, pb_dim)
        with torch.no_grad():
            for object_id in range(num_objects):
                self.material_property.weight[object_id].fill_(object_id * 0.2)
        mlp_input_dim = (
            n_obs_steps * (image_feature_dim + state_dim)
            + self.n_action_condition_steps * action_dim
            + pb_dim
        )
        layers = []
        for _ in range(3):
            layers.extend(
                [
                    nn.Linear(mlp_input_dim, dim_feedforward),
                    nn.LayerNorm(dim_feedforward),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            mlp_input_dim = dim_feedforward
        layers.append(nn.Linear(dim_feedforward, horizon * self.trajectory_dim))
        self.output_mlp = nn.Sequential(*layers)

        print(
            "WrenchPredictor4MLP params: %e" % sum(p.numel() for p in self.parameters())
        )
        print(
            "Material PB params: %e"
            % sum(p.numel() for p in self.material_property.parameters())
        )

    def get_condition_input(self, batch, material_property=None):
        image_feature = batch["image_feature"][:, : self.n_obs_steps]
        state = batch["state"][:, : self.n_obs_steps]
        action = batch["action"][:, self.n_obs_steps - 1 : self.horizon - 1]
        if material_property is None:
            material_property = self.material_property(batch["object_id"])

        return torch.cat(
            [
                image_feature.flatten(start_dim=1),
                state.flatten(start_dim=1),
                action.flatten(start_dim=1),
                material_property,
            ],
            dim=1,
        )

    def forward(self, batch, material_property=None):
        condition_input = self.get_condition_input(batch, material_property)
        trajectory = self.output_mlp(condition_input).reshape(
            condition_input.shape[0],
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
