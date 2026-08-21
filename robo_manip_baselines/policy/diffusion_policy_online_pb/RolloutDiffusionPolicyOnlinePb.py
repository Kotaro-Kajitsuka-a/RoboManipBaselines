from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from robo_manip_baselines.common import (
    DataKey,
    convert_data_to_policy,
    normalize_data,
)
from robo_manip_baselines.policy.diffusion_policy import RolloutDiffusionPolicy
from robo_manip_baselines.policy.wrench_predictor4_online.WrenchPredictor4OnlineUtils import (
    ONLINE_PB_STD_KEY,
    GaussianBeliefOnlinePb,
    calculate_pb_candidate_losses,
    load_model_meta_info,
    load_pb,
    load_policy,
    resolve_gaussian_num_points,
)


class RolloutDiffusionPolicyOnlinePb(RolloutDiffusionPolicy):
    """Roll out Diffusion Policy while adapting a standalone WP4 PB online."""

    def set_additional_args(self, parser):
        super().set_additional_args(parser)
        parser.add_argument(
            "--wp4_checkpoint",
            type=Path,
            required=True,
            help="WrenchPredictor4 checkpoint used for online PB adaptation",
        )
        parser.add_argument(
            "--initial_object_id",
            type=int,
            default=0,
            help=(
                "trained WP4 object PB used at the start of every episode; "
                "valid IDs are read from the WP4 checkpoint"
            ),
        )
        parser.add_argument(
            "--fixed_pb",
            action="store_true",
            help=(
                "keep the PB selected by --initial_object_id fixed throughout "
                "the rollout instead of adapting it online"
            ),
        )
        parser.add_argument(
            "--online_pb_lr",
            type=float,
            default=6e-3,
            help="learning rate applied only to the online PB",
        )
        parser.add_argument(
            "--online_pb_update_type",
            choices=("adam", "gaussian_belief"),
            default="adam",
            help="online PB update rule",
        )
        parser.add_argument(
            "--online_pb_initial_std",
            type=float,
            default=None,
            help="initial PB standard deviation; required for gaussian_belief",
        )
        parser.add_argument(
            "--online_pb_num_points",
            type=int,
            default=None,
            help="Gauss-Hermite order (default: 16 times the PB dimension)",
        )
        parser.add_argument(
            "--online_pb_beta",
            type=float,
            default=1.0,
            help="pseudo-likelihood inverse temperature for gaussian_belief",
        )
        parser.add_argument(
            "--wrench_loss_weight",
            type=float,
            default=0.0,
            help="weight of the normalized wrench prediction loss used to adapt PB",
        )
        parser.add_argument(
            "--image_vae_checkpoint_path",
            "--image_vae_checkpoint",
            type=Path,
            default=None,
            help="image VAE used when WP4 predicts an offline VAE feature",
        )
        parser.add_argument(
            "--image_vae_camera_name",
            type=str,
            default="hand",
            help="camera encoded by --image_vae_checkpoint_path",
        )

    def setup_policy(self):
        super().setup_policy()

        assert self.state_keys.count(DataKey.MATERIAL_PROPERTY) == 1, self.state_keys
        assert self.args.wrench_loss_weight >= 0.0, self.args.wrench_loss_weight
        if self.args.online_pb_update_type == "adam":
            assert self.args.online_pb_lr > 0.0, self.args.online_pb_lr
        else:
            assert (
                self.args.online_pb_initial_std is not None
                and self.args.online_pb_initial_std > 0.0
            ), self.args.online_pb_initial_std
            if self.args.online_pb_num_points is not None:
                assert (
                    self.args.online_pb_num_points >= 3
                ), self.args.online_pb_num_points
            assert self.args.online_pb_beta > 0.0, self.args.online_pb_beta
        self.wp4_checkpoint = self.args.wp4_checkpoint.resolve()
        self.wp4_model_meta_info = load_model_meta_info(self.wp4_checkpoint)
        self.wp4_policy = load_policy(
            self.wp4_checkpoint,
            self.wp4_model_meta_info,
            self.device,
        )
        self.initial_pb, self.initial_object_key = load_pb(
            self.wp4_checkpoint,
            self.args.initial_object_id,
            self.wp4_model_meta_info,
        )

        non_pb_state_dim = sum(
            DataKey.get_dim_for_policy(state_key, self.env)
            for state_key in self.state_keys
            if state_key != DataKey.MATERIAL_PROPERTY
        )
        dp_pb_dim = self.state_dim - non_pb_state_dim
        wp4_pb_dim = self.wp4_model_meta_info["material_property"]["pb_dim"]
        assert dp_pb_dim == wp4_pb_dim, (dp_pb_dim, wp4_pb_dim)
        self.online_pb_num_points = resolve_gaussian_num_points(
            wp4_pb_dim,
            self.args.online_pb_num_points,
        )

        wrench_clip_info = self.wp4_model_meta_info["wrench"]["percentile_clip"]
        self.wp4_wrench_source_key = wrench_clip_info["source_key"]
        self.wp4_wrench_clip_min = np.asarray(wrench_clip_info["min"])
        self.wp4_wrench_clip_max = np.asarray(wrench_clip_info["max"])

        self.wp4_image_vae = None
        self.wp4_image_feature_key = self.wp4_model_meta_info["data"][
            "image_feature_key"
        ]
        if self.wp4_image_feature_key.startswith("image_vae"):
            assert self.args.image_vae_checkpoint_path is not None
            from pythae.models import AutoModel

            self.image_vae_checkpoint_path = (
                self.args.image_vae_checkpoint_path.resolve()
            )
            self.wp4_image_vae = AutoModel.load_from_folder(
                str(self.image_vae_checkpoint_path)
            )
            self.wp4_image_vae.eval().to(self.device)
            self.wp4_image_vae.requires_grad_(False)
            input_dim = self.wp4_image_vae.model_config.input_dim
            assert self.wp4_image_vae.model_config.latent_dim == len(
                self.wp4_model_meta_info["image_feature"]["example"]
            )
            self.wp4_image_size = (input_dim[2], input_dim[1])

        wp4_data_info = self.wp4_model_meta_info["data"]
        print(
            f"[{self.__class__.__name__}] Construct online PB adapter.\n"
            f"  - WP4 checkpoint: {self.wp4_checkpoint}\n"
            f"  - initial object: {self.initial_object_key} "
            f"(id={self.args.initial_object_id}, PB={self.initial_pb.tolist()})\n"
            f"  - fixed PB: {self.args.fixed_pb}\n"
            f"  - PB dimension: {wp4_pb_dim}\n"
            f"  - update type: {self.args.online_pb_update_type}\n"
            f"  - wrench loss weight: {self.args.wrench_loss_weight}\n"
            f"  - horizon: {wp4_data_info['horizon']}, "
            f"obs steps: {wp4_data_info['n_obs_steps']}, "
            f"skip: {wp4_data_info['skip']}"
        )
        if self.args.online_pb_update_type == "adam":
            print(f"  - learning rate: {self.args.online_pb_lr}")
        else:
            print(
                f"  - initial std: {self.args.online_pb_initial_std}\n"
                f"  - Gauss-Hermite points: {self.online_pb_num_points}\n"
                f"  - pseudo-likelihood beta: {self.args.online_pb_beta}\n"
                "  - update interval: every window (overlapping evidence)"
            )
        if self.wp4_image_vae is not None:
            print(
                f"  - image feature: {self.wp4_image_feature_key} from "
                f"{self.args.image_vae_camera_name} camera, model: "
                f"{self.image_vae_checkpoint_path}"
            )

    def setup_variables(self):
        super().setup_variables()
        self.data_manager.meta_data["online_pb_wp4_checkpoint"] = str(
            self.wp4_checkpoint
        )
        self.data_manager.meta_data["online_pb_initial_object_id"] = (
            self.args.initial_object_id
        )
        self.data_manager.meta_data["online_pb_fixed"] = self.args.fixed_pb
        self.data_manager.meta_data["online_pb_update_type"] = (
            self.args.online_pb_update_type
        )
        self.data_manager.meta_data["online_pb_wrench_loss_weight"] = (
            self.args.wrench_loss_weight
        )
        if self.args.online_pb_update_type == "adam":
            self.data_manager.meta_data["online_pb_learning_rate"] = (
                self.args.online_pb_lr
            )
        else:
            self.data_manager.meta_data["online_pb_initial_std"] = (
                self.args.online_pb_initial_std
            )
            self.data_manager.meta_data["online_pb_gauss_hermite_num_points"] = (
                self.online_pb_num_points
            )
            self.data_manager.meta_data["online_pb_pseudo_likelihood_beta"] = (
                self.args.online_pb_beta
            )
            self.data_manager.meta_data["online_pb_overlapping_evidence"] = True
        if self.wp4_image_vae is not None:
            self.data_manager.meta_data["online_pb_image_vae_checkpoint"] = str(
                self.image_vae_checkpoint_path
            )
            self.data_manager.meta_data["online_pb_image_vae_camera_name"] = (
                self.args.image_vae_camera_name
            )

    def reset_variables(self):
        super().reset_variables()

        horizon = self.wp4_model_meta_info["data"]["horizon"]
        self.online_observation_window = deque(maxlen=horizon)
        if self.args.online_pb_update_type == "adam":
            self.online_pb = torch.nn.Parameter(
                torch.tensor(
                    self.initial_pb,
                    dtype=torch.float32,
                    device=self.device,
                )
            )
            self.online_pb_optimizer = torch.optim.Adam(
                [self.online_pb],
                lr=self.args.online_pb_lr,
            )
            self.online_pb_belief = None
        else:
            self.online_pb_belief = GaussianBeliefOnlinePb(
                self.initial_pb,
                self.args.online_pb_initial_std,
                self.online_pb_num_points,
                self.args.online_pb_beta,
                self.device,
            )
            self.online_pb = self.online_pb_belief.mean
            self.online_pb_optimizer = None

    def infer_policy(self):
        if not self.args.fixed_pb:
            self.append_online_observation()
            if (
                len(self.online_observation_window)
                == self.online_observation_window.maxlen
            ):
                # RolloutPhase calls infer_policy() under torch.inference_mode().
                # Re-enable normal tensor creation locally for either update rule.
                with torch.inference_mode(False):
                    if self.args.online_pb_update_type == "adam":
                        with torch.enable_grad():
                            self.update_online_pb()
                    else:
                        with torch.no_grad():
                            self.update_online_pb()

        # The updated PB is inserted into the DP state by update_state_buf().
        # Keep the existing DP action buffer; the latest PB takes effect at the
        # next normal Diffusion Policy inference.
        super().infer_policy()

    def append_online_observation(self):
        state = np.concatenate(
            [
                convert_data_to_policy(
                    self.motion_manager.get_data(state_key, self.obs),
                    state_key,
                )
                for state_key in self.wp4_model_meta_info["state"]["keys"]
            ]
        )
        if self.wp4_image_vae is None:
            image_feature = convert_data_to_policy(
                self.motion_manager.get_data(self.wp4_image_feature_key, self.obs),
                self.wp4_image_feature_key,
            )
        else:
            image_feature = self.encode_wp4_image_feature()
        wrench = np.clip(
            self.motion_manager.get_data(self.wp4_wrench_source_key, self.obs),
            self.wp4_wrench_clip_min,
            self.wp4_wrench_clip_max,
        )
        self.online_observation_window.append(
            {
                "state": state.copy(),
                "image_feature": image_feature.copy(),
                "wrench": wrench.copy(),
            }
        )

    def encode_wp4_image_feature(self):
        image = self.info["rgb_images"][self.args.image_vae_camera_name]
        image = cv2.resize(
            image,
            self.wp4_image_size,
            interpolation=cv2.INTER_LINEAR,
        )
        image = torch.from_numpy(image).to(self.device)
        image = image.permute(2, 0, 1).unsqueeze(0).float() / 255.0
        with torch.no_grad():
            feature = self.wp4_image_vae.encoder(image).embedding[0]
        return feature.cpu().numpy()

    def update_online_pb(self):
        horizon = self.wp4_model_meta_info["data"]["horizon"]
        assert (
            len(self.policy_action_list) >= horizon - 1
        ), (  # if horizon = 16, we need at least excuted 15 actions.
            len(self.policy_action_list),
            horizon,
        )

        state = np.stack([sample["state"] for sample in self.online_observation_window])
        image_feature = np.stack(
            [sample["image_feature"] for sample in self.online_observation_window]
        )
        wrench = np.stack(
            [sample["wrench"] for sample in self.online_observation_window]
        )

        # At observation t, actions through t - 1 have already been executed.
        # WP4 consumes action indices [n_obs_steps - 1, horizon - 1), so the
        # final action slot is unused and can safely hold the latest past action.
        past_action = self.policy_action_list[-(horizon - 1) :].copy()
        dummy_action = np.zeros_like(past_action[-1:])
        action = np.concatenate([past_action, dummy_action], axis=0)

        batch = {
            "state": torch.tensor(
                normalize_data(state, self.wp4_model_meta_info["state"]),
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0),
            "action": torch.tensor(
                normalize_data(action, self.wp4_model_meta_info["action"]),
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0),
            "image_feature": torch.tensor(
                normalize_data(
                    image_feature,
                    self.wp4_model_meta_info["image_feature"],
                ),
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0),
            "wrench": torch.tensor(
                normalize_data(wrench, self.wp4_model_meta_info["wrench"]),
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0),
        }

        if self.args.online_pb_update_type == "adam":
            self.online_pb_optimizer.zero_grad()
            prediction = self.wp4_policy(batch, self.online_pb.unsqueeze(0))
            start = self.wp4_policy.n_obs_steps
            pose_loss = F.mse_loss(
                prediction["image_feature"][:, start:],
                batch["image_feature"][:, start:],
            )
            wrench_loss = F.mse_loss(
                prediction["wrench"][:, start:],
                batch["wrench"][:, start:],
            )
            loss = pose_loss + self.args.wrench_loss_weight * wrench_loss
            loss.backward()
            self.online_pb_optimizer.step()
        else:
            losses = calculate_pb_candidate_losses(
                self.wp4_policy,
                batch,
                self.online_pb_belief.get_candidates(),
                self.args.wrench_loss_weight,
            )
            self.online_pb_belief.update(losses)

    # Override methods in normal Diffusion Policy to use the online PB in the DP state buffer.
    def update_state_buf(self):
        state = np.concatenate(
            [
                self.get_dp_state_data_including_pb(state_key)
                for state_key in self.state_keys
            ]
        )
        state = normalize_data(state, self.model_meta_info["state"])
        state = torch.tensor(state, dtype=torch.float32)

        if self.state_buf is None:
            self.state_buf = [
                state for _ in range(self.model_meta_info["data"]["n_obs_steps"])
            ]
        else:
            self.state_buf.pop(0)
            self.state_buf.append(state)

    def get_dp_state_data_including_pb(self, state_key):
        if state_key == DataKey.MATERIAL_PROPERTY:
            return self.online_pb.detach().cpu().numpy().copy()
        return convert_data_to_policy(
            self.motion_manager.get_data(state_key, self.obs),
            state_key,
        )

    def record_data(self):
        super().record_data()
        self.data_manager.append_single_data(
            DataKey.MATERIAL_PROPERTY,
            self.online_pb.detach().cpu().numpy().copy(),
        )
        if self.online_pb_belief is not None:
            self.data_manager.append_single_data(
                ONLINE_PB_STD_KEY,
                self.online_pb_belief.std.detach().cpu().numpy().copy(),
            )
