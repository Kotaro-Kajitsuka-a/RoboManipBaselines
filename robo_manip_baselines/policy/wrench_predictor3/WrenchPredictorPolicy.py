import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms

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
            material_property_dim=material_property_dim,
            horizon=self.policy_args["horizon"],
            camera_names=self.policy_args["camera_names"],
            image_shape=self.policy_args["image_shape"],
            hidden_dim=self.policy_args["hidden_dim"],
            dim_feedforward=self.policy_args["dim_feedforward"],
            enc_layers=self.policy_args["enc_layers"],
            nheads=self.policy_args["nheads"],
        )

        n_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("number of parameters: %.2fM" % (n_parameters / 1e6,))

    def forward(
        self,
        state,
        image,
        material_object_id=None,
        wrench=None,
        material_property=None,
    ):
        # If material_property is given, use it.
        # Otherwise, use material_object_id to get the material property embedding.
        if material_property is None:
            assert material_object_id is not None, (
                "Either material_object_id or material_property must be provided."
            )
            material_property = self.material_property_embedding(material_object_id)
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        image = normalize(image)
        if wrench is not None:  # training time
            wrench_hat = self.model(state, image, material_property)
            loss_dict = dict()
            loss_dict["l1"] = F.l1_loss(wrench, wrench_hat)
            loss_dict["loss"] = loss_dict["l1"]
            return loss_dict
        else:  # inference time
            return self.model(state, image, material_property)

    def configure_optimizers(self):
        param_dicts = []
        material_property_params = [
            p
            for n, p in self.named_parameters()
            if n.startswith("material_property_embedding") and p.requires_grad
        ]
        if len(material_property_params) > 0:
            param_dicts.append(
                {
                    "params": material_property_params,
                    "lr": self.policy_args["lr_material_property"],
                }
            )

        model_params = [
            p
            for n, p in self.named_parameters()
            if not n.startswith("material_property_embedding")
            and not n.startswith("model.cnn")
            and p.requires_grad
        ]
        if len(model_params) > 0:
            param_dicts.append(
                {
                    "params": model_params,
                    "lr": self.policy_args["lr"],
                }
            )

        backbone_params = [
            p
            for n, p in self.named_parameters()
            if n.startswith("model.cnn") and p.requires_grad
        ]
        if len(backbone_params) > 0:
            param_dicts.append(
                {
                    "params": backbone_params,
                    "lr": self.policy_args["lr_backbone"],
                }
            )
        assert len(param_dicts) > 0, "No trainable WrenchPredictor3 parameters."
        return torch.optim.AdamW(
            param_dicts,
            lr=self.policy_args["lr"],
            weight_decay=self.policy_args["weight_decay"],
        )
