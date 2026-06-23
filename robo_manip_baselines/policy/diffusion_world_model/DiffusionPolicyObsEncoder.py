import os
import pickle
import sys

import numpy as np
import torch

from robo_manip_baselines.common import DataKey

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
        self.feature_slices = self.get_feature_slices()

    @staticmethod
    def load_model_meta_info(checkpoint_path):
        checkpoint_dir = os.path.dirname(checkpoint_path)
        model_meta_info_path = os.path.join(checkpoint_dir, "model_meta_info.pkl")
        with open(model_meta_info_path, "rb") as f:
            return pickle.load(f)

    def load_obs_encoder(self):
        policy = self._construct_policy()
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

    def _construct_policy(self):
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

    @torch.inference_mode()
    def encode_visual(self, obs_dict):
        feature = self.encode(obs_dict)
        visual_feature_list = [
            feature[:, feature_slice]
            for key, feature_slice in self.feature_slices.items()
            if DataKey.is_rgb_image_key(key)
        ]
        assert len(visual_feature_list) > 0, self.feature_slices
        return torch.cat(visual_feature_list, dim=-1)

    @torch.no_grad()
    def output_shape(self):
        return self.obs_encoder.output_shape()

    def visual_output_shape(self):
        dim = 0
        for key, feature_slice in self.feature_slices.items():
            if DataKey.is_rgb_image_key(key):
                dim += feature_slice.stop - feature_slice.start
        return [dim]

    def get_feature_slices(self):
        feature_slices = {}
        start = 0
        for key in self.obs_encoder.obs_shapes:
            feature_shape = self.obs_encoder.obs_shapes[key]

            randomizer = self.obs_encoder.obs_randomizers[key]
            if randomizer is not None:
                feature_shape = randomizer.output_shape_in(feature_shape)

            obs_net = self.obs_encoder.obs_nets[key]
            if obs_net is not None:
                feature_shape = obs_net.output_shape(feature_shape)

            if randomizer is not None:
                feature_shape = randomizer.output_shape_out(feature_shape)

            dim = int(np.prod(feature_shape))
            feature_slices[key] = slice(start, start + dim)
            start += dim

        assert start == self.obs_encoder.output_shape()[0], (
            start,
            self.obs_encoder.output_shape(),
        )
        return feature_slices
