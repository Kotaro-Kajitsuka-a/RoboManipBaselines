import os
import pickle
import sys

import torch

sys.path.append(
    os.path.join(os.path.dirname(__file__), "../../../third_party/diffusion_policy")
)


class FrozenDiffusionPolicyObsEncoder:
    """Frozen observation encoder extracted from a trained Diffusion Policy."""

    def __init__(self, checkpoint_path, device="cuda"):
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device)

        self.model_meta_info = self.load_model_meta_info(checkpoint_path)
        self.obs_encoder = self.load_obs_encoder()
        self.obs_encoder.eval()
        self.obs_encoder.requires_grad_(False)

    @staticmethod
    def load_model_meta_info(checkpoint_path):
        checkpoint_dir = os.path.dirname(checkpoint_path)
        model_meta_info_path = os.path.join(checkpoint_dir, "model_meta_info.pkl")
        with open(model_meta_info_path, "rb") as f:
            return pickle.load(f)

    def load_obs_encoder(self):
        policy = self.construct_policy()
        policy.load_state_dict(
            torch.load(
                self.checkpoint_path,
                map_location=self.device,
                weights_only=True,
            )
        )
        policy.to(self.device)

        obs_encoder = policy.obs_encoder
        del policy
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return obs_encoder

    def construct_policy(self):
        if "backbone" not in self.model_meta_info["policy"]:
            self.model_meta_info["policy"]["backbone"] = "cnn"
        if "scheduler" not in self.model_meta_info["policy"]:
            self.model_meta_info["policy"]["scheduler"] = "ddpm"

        if self.model_meta_info["policy"]["scheduler"] == "ddpm":
            from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

            noise_scheduler = DDPMScheduler(
                **self.model_meta_info["policy"]["noise_scheduler_args"]
            )
        elif self.model_meta_info["policy"]["scheduler"] == "ddim":
            from diffusers.schedulers.scheduling_ddim import DDIMScheduler

            noise_scheduler = DDIMScheduler(
                **self.model_meta_info["policy"]["noise_scheduler_args"]
            )
        else:
            raise ValueError(
                f"Invalid scheduler: {self.model_meta_info['policy']['scheduler']}"
            )

        if self.model_meta_info["policy"]["backbone"] == "cnn":
            from diffusion_policy.policy.diffusion_unet_hybrid_image_policy import (
                DiffusionUnetHybridImagePolicy,
            )

            PolicyClass = DiffusionUnetHybridImagePolicy
        else:
            raise ValueError(
                f"Invalid backbone: {self.model_meta_info['policy']['backbone']}"
            )

        return PolicyClass(
            noise_scheduler=noise_scheduler,
            **self.model_meta_info["policy"]["args"],
        )

    @torch.inference_mode()
    def encode(self, obs_dict):
        return self.obs_encoder(obs_dict)

    @torch.no_grad()
    def output_shape(self):
        return self.obs_encoder.output_shape()
