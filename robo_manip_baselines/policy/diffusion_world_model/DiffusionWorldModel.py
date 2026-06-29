import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(
    os.path.join(os.path.dirname(__file__), "../../../third_party/diffusion_policy")
)
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D


class DiffusionWorldModel(nn.Module):
    """CNN-DiffusionPolicy style world model.

    The denoising trajectory is [wrench, image_feature].
    The global condition is [observed image_feature/state, planned action, material PB].
    """

    def __init__(
        self,
        noise_scheduler,
        image_feature_dim,
        state_dim,
        action_dim,
        wrench_dim,
        num_objects,
        pb_dim,
        horizon,
        n_obs_steps,
        num_inference_steps=None,
        diffusion_step_embed_dim=128,
        down_dims=(256, 512, 1024),
        kernel_size=5,
        n_groups=8,
        cond_predict_scale=True,
        image_feature_target_mode="absolute",
    ):
        super().__init__()

        self.noise_scheduler = noise_scheduler
        self.material_property = nn.Embedding(num_objects, pb_dim)

        self.image_feature_dim = image_feature_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.wrench_dim = wrench_dim
        self.trajectory_dim = wrench_dim + image_feature_dim
        self.pb_dim = pb_dim
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.image_feature_target_mode = image_feature_target_mode
        self.n_action_condition_steps = horizon - n_obs_steps
        assert self.n_action_condition_steps > 0, (horizon, n_obs_steps)
        assert self.image_feature_target_mode in (
            "absolute",
            "delta_from_last_obs",
        ), self.image_feature_target_mode

        global_cond_dim = (
            n_obs_steps * (image_feature_dim + state_dim)
            + self.n_action_condition_steps * action_dim
            + pb_dim
        )
        self.model = ConditionalUnet1D(
            input_dim=self.trajectory_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
        )

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        print("Diffusion params: %e" % sum(p.numel() for p in self.model.parameters()))
        print(
            "Material PB params: %e"
            % sum(p.numel() for p in self.material_property.parameters())
        )

    # ========= inference  ============
    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def get_trajectory(self, batch):
        return torch.cat(
            [batch["wrench"], self.get_image_feature_target(batch)],
            dim=-1,
        )

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

    # ========= training  ============
    def get_global_cond(self, batch):
        image_feature = batch["image_feature"][:, : self.n_obs_steps]
        state = batch["state"][:, : self.n_obs_steps]
        action = batch["action"][:, self.n_obs_steps - 1 : self.horizon - 1]
        material_property = self.material_property(batch["object_id"])

        return torch.cat(
            [
                image_feature.flatten(start_dim=1),
                state.flatten(start_dim=1),
                action.flatten(start_dim=1),
                material_property,
            ],
            dim=-1,
        )

    def compute_loss(self, batch):
        trajectory = self.get_trajectory(batch)
        global_cond = self.get_global_cond(batch)

        # Sample noise that we'll add to the trajectory
        noise = torch.randn(
            trajectory.shape, device=trajectory.device, dtype=trajectory.dtype
        )
        # Sample a random timestep for each sample
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (trajectory.shape[0],),
            device=trajectory.device,
        ).long()
        # Add noise to the clean trajectory according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)

        # Predict the noise residual
        pred = self.model(noisy_trajectory, timesteps, global_cond=global_cond)
        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == "epsilon":
            target = noise
        elif pred_type == "sample":
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction="none")
        loss = loss.mean()
        return loss

    @torch.no_grad()
    def predict(self, batch):
        global_cond = self.get_global_cond(batch)
        sample = torch.randn(
            (global_cond.shape[0], self.horizon, self.trajectory_dim),
            device=self.device,
            dtype=self.dtype,
        )

        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        for timestep in self.noise_scheduler.timesteps:
            model_output = self.model(sample, timestep, global_cond=global_cond)
            sample = self.noise_scheduler.step(
                model_output, timestep, sample
            ).prev_sample

        wrench = sample[..., : self.wrench_dim]
        image_feature_target = sample[..., self.wrench_dim :]
        image_feature = self.restore_image_feature(batch, image_feature_target)
        return {"wrench": wrench, "image_feature": image_feature}
