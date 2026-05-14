import torch
import torch.nn as nn
from torch.nn import functional as F

from .WrenchPredictorModel import WrenchPredictorModel


class WrenchPredictorPolicy(nn.Module):
    def __init__(
        self,
        state_dim,
        wrench_dim,
        num_material_objects,
        material_property_dim,
        policy_args,
    ):
        super().__init__()
        self.policy_args = policy_args
        self.material_property_embedding = nn.Embedding(
            num_material_objects,
            material_property_dim,
        )
        nn.init.zeros_(self.material_property_embedding.weight)
        self.model = WrenchPredictorModel(
            state_dim=state_dim,
            wrench_dim=wrench_dim,
            chunk_size=self.policy_args["chunk_size"],
            hidden_dim=self.policy_args["hidden_dim"],
            dim_feedforward=self.policy_args["dim_feedforward"],
        )

        n_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("number of parameters: %.2fM" % (n_parameters / 1e6,))

    def forward(
        self,
        state,
        material_object_id=None,
        wrench=None,
        is_pad=None,
        material_property=None,
    ):
        if material_property is None:
            assert material_object_id is not None
            material_property = self.material_property_embedding(material_object_id)

        if wrench is not None:
            assert is_pad is not None
            wrench = wrench[:, : self.model.chunk_size]
            is_pad = is_pad[:, : self.model.chunk_size]

            wrench_hat = self.model(state, material_property)
            loss_dict = {}
            all_l1 = F.l1_loss(wrench, wrench_hat, reduction="none")
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict["l1"] = l1
            loss_dict["loss"] = loss_dict["l1"]
            return loss_dict

        return self.model(state, material_property)

    def configure_optimizers(self):
        param_dicts = [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if n.startswith("material_property_embedding") and p.requires_grad
                ],
                "lr": self.policy_args["lr_material_property"],
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not n.startswith("material_property_embedding")
                    and p.requires_grad
                ]
            },
        ]
        return torch.optim.AdamW(
            param_dicts,
            lr=self.policy_args["lr"],
            weight_decay=self.policy_args["weight_decay"],
        )
