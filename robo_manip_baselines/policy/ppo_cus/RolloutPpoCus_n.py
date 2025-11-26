import cv2
import matplotlib.pylab as plt
import numpy as np
import torch

from robo_manip_baselines.common import RolloutBase, denormalize_data, normalize_data

from .MlpPolicy import MlpPolicy


class RolloutMlp(RolloutBase):
    def setup_policy(self):
        # Print policy information
        self.print_policy_info()
        print(
            f"  - obs steps: {self.model_meta_info['data']['n_obs_steps']}, action steps: {self.model_meta_info['data']['n_action_steps']}"
        )

        # Construct policy
        self.policy = MlpPolicy(
            self.state_dim,
            self.action_dim,
            len(self.camera_names),
            **self.model_meta_info["policy"]["args"],
        )

        # Load checkpoint
        self.load_ckpt()

    def setup_policy(self):
        # Always keep vision enabled for marker detection

        self.print_policy_info()
        print(
            f"  - obs steps: {self.model_meta_info['data']['n_obs_steps']}, action steps: {self.model_meta_info['data']['n_action_steps']}"
        )

        state_dict = torch.load(self.args.checkpoint, map_location="cpu")

        if "actor_mean.0.weight" not in state_dict or "actor_logstd" not in state_dict:
            raise KeyError(
                f"[{self.__class__.__name__}] ManiSkill PPO checkpoint does not contain expected keys."
            )

        obs_dim = state_dict["actor_mean.0.weight"].shape[1]
        action_dim_from_ckpt = int(state_dict["actor_logstd"].shape[-1])
        if action_dim_from_ckpt != self.action_dim:
            raise ValueError(
                f"[{self.__class__.__name__}] action dim mismatch: meta={self.action_dim}, checkpoint={action_dim_from_ckpt}"
            )

        self.policy = ManiSkillPpoAgent(obs_dim, self.action_dim)
        self.policy.load_state_dict(state_dict)

        use_cuda = _PPO_USE_CUDA and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.policy.to(self.device)
        self.policy.eval()
        self._policy_obs_dim = obs_dim

        self._normalized_action_low = torch.full(
            (self.action_dim,), float(_NORMALIZED_ACTION_LOW.item()), device=self.device
        )
        self._normalized_action_high = torch.full(
            (self.action_dim,), float(_NORMALIZED_ACTION_HIGH.item()), device=self.device
        )
        self._init_action_scaling_tensors()

        print(
            f"[{self.__class__.__name__}] Load ManiSkill PPO checkpoint on {self.device}"
        )

        expanded_path = _REPO_ROOT / "robo_manip_baselines" / "rollout_debug_log_expanded.csv"
        header = ["step_idx"]
        header += [f"obs_{idx}" for idx in range(self._policy_obs_dim)]
        header += [f"raw_action_{idx}" for idx in range(self.action_dim)]
        header += [f"direct_joint_command_{idx}" for idx in range(self.action_dim)]
        with open(expanded_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
        self._expanded_csv_path: Optional[Path] = expanded_path
        print(
            f"[{self.__class__.__name__}] Logging expanded observations/actions to {expanded_path}"
        )

        self._log_path = None
        if _PPO_LOG_TSV:
            checkpoint_dir = os.path.dirname(os.path.abspath(self.args.checkpoint))
            default_name = f"{self.__class__.__name__.lower()}_debug_log.tsv"
            self._log_path = os.path.join(checkpoint_dir, default_name)
            os.makedirs(os.path.dirname(self._log_path), exist_ok=True)
            with open(self._log_path, "w", newline="") as f:
                writer = csv.writer(f, delimiter="\t")
                writer.writerow(["step_idx", "obs", "direct_joint_command"])
            print(
                f"[{self.__class__.__name__}] Logging observations and actions to {self._log_path}"
            )

    def setup_plot(self):
        fig_ax = plt.subplots(
            2,
            len(self.camera_names),
            figsize=(13.5, 6.0),
            dpi=60,
            squeeze=False,
            constrained_layout=True,
        )
        super().setup_plot(fig_ax)

    def reset_variables(self):
        super().reset_variables()

        self.state_buf = None
        self.images_buf = None
        self.policy_action_buf = None

    def update_state_buf(self):
        if len(self.state_keys) == 0:
            state = np.zeros(0, dtype=np.float32)
        else:
            state = np.concatenate(
                [
                    self.motion_manager.get_data(state_key, self.obs)
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

    def get_state(self):
        return torch.stack(self.state_buf, dim=0)[torch.newaxis].to(self.device)

    def update_images_buf(self):
        images = []
        for camera_name in self.camera_names:
            image = self.info["rgb_images"][camera_name]

            image = np.moveaxis(image, -1, -3)
            image = torch.tensor(image.copy(), dtype=torch.uint8)
            image = self.image_transforms(image)

            images.append(image)

        if self.images_buf is None:
            self.images_buf = [
                [image for _ in range(self.model_meta_info["data"]["n_obs_steps"])]
                for image in images
            ]
        else:
            for single_images_buf, image in zip(self.images_buf, images):
                single_images_buf.pop(0)
                single_images_buf.append(image)

    def get_images(self):
        return torch.stack(
            [
                torch.stack(single_images_buf, dim=0)[torch.newaxis].to(self.device)
                for single_images_buf in self.images_buf
            ]
        )

    def infer_policy(self):
        # Update observation buffer
        self.update_state_buf()
        self.update_images_buf()

        # Infer
        if self.policy_action_buf is None or len(self.policy_action_buf) == 0:
            state = self.get_state()
            images = self.get_images()
            action = self.policy(state, images)[0]
            self.policy_action_buf = list(
                action.cpu().detach().numpy().astype(np.float64)
            )

        # Store action
        self.policy_action = denormalize_data(
            self.policy_action_buf.pop(0), self.model_meta_info["action"]
        )
        self.policy_action_list = np.concatenate(
            [self.policy_action_list, self.policy_action[np.newaxis]]
        )

    def draw_plot(self):
        # Clear plot
        for _ax in np.ravel(self.ax):
            _ax.cla()
            _ax.axis("off")

        # Plot images
        self.plot_images(self.ax[0, 0 : len(self.camera_names)])

        # Plot action
        self.plot_action(self.ax[1, 0])

        # Finalize plot
        self.canvas.draw()
        cv2.imshow(
            self.policy_name,
            cv2.cvtColor(np.asarray(self.canvas.buffer_rgba()), cv2.COLOR_RGB2BGR),
        )
