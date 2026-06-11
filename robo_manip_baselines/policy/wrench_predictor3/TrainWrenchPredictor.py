import numpy as np
import torch
from tqdm import tqdm

from robo_manip_baselines.common import TrainBase
from robo_manip_baselines.common.data.DataKey import DataKey
from robo_manip_baselines.common.data.RmbData import RmbData
from robo_manip_baselines.misc.AddPercentileClippedWrenchToRmbData import (
    AddPercentileClippedWrenchToRmbData,
)

from .MaterialPropertyUtils import build_material_object_key_to_id
from .WrenchPredictorDataset import WrenchPredictorDataset
from .WrenchPredictorPolicy import WrenchPredictorPolicy
from .WrenchPredictorSequenceUtils import build_condition, build_wrench_target


class TrainWrenchPredictor(TrainBase):
    DatasetClass = WrenchPredictorDataset

    def set_additional_args(self, parser):
        parser.set_defaults(enable_rmb_cache=True)

        parser.set_defaults(image_aug_std=0.1)

        parser.set_defaults(batch_size=64)
        parser.set_defaults(num_epochs=1000)
        parser.set_defaults(lr=1e-5)

        parser.add_argument(
            "--hidden_dim", type=int, default=512, help="hidden dimension"
        )
        parser.add_argument(
            "--dim_feedforward", type=int, default=2048, help="feedforward dimension"
        )
        parser.add_argument(
            "--image_width", type=int, default=320, help="input image width"
        )
        parser.add_argument(
            "--image_height", type=int, default=240, help="input image height"
        )
        parser.add_argument(
            "--lr_material_property",
            type=float,
            default=1e-3,
            help="learning rate for material property embedding; set 0 to freeze it",
        )
        parser.add_argument(
            "--lr_backbone",
            type=float,
            default=1e-5,
            help="learning rate for image CNN backbone; set 0 to freeze it",
        )

    def setup_model_meta_info(self):
        super().setup_model_meta_info()

        self.model_meta_info["data"]["image_size"] = (
            self.args.image_width,
            self.args.image_height,
        )
        self.model_meta_info["data"]["horizon"] = self.args.skip
        self.model_meta_info["material_property"] = {
            "dim": 9,
            "object_key_to_id": build_material_object_key_to_id(self.all_filenames),
        }

    def set_data_stats(self):
        AddPercentileClippedWrenchToRmbData(
            self.args.dataset_dir,
            overwrite=True,
        ).run()

        all_state = []
        all_wrench = []
        rgb_image_example = None
        depth_image_example = None
        episode_len_list = []
        horizon = self.args.skip
        clip_min = None
        clip_max = None
        for filename in self.all_filenames:
            with RmbData(
                filename, image_size=self.model_meta_info["data"]["image_size"]
            ) as rmb_data:
                episode_len = rmb_data[DataKey.TIME].shape[0]
                episode_len_list.append(episode_len)
                try:
                    file_clip_min = np.asarray(
                        rmb_data.attrs[
                            DataKey.MEASURED_EEF_WRENCH_MOVING_AVERAGE_PERCENTILE_CLIP
                            + "_clip_min"
                        ],
                        dtype=np.float64,
                    )
                    file_clip_max = np.asarray(
                        rmb_data.attrs[
                            DataKey.MEASURED_EEF_WRENCH_MOVING_AVERAGE_PERCENTILE_CLIP
                            + "_clip_max"
                        ],
                        dtype=np.float64,
                    )
                except KeyError as e:
                    raise KeyError(f"{e}: {filename}") from e
                if clip_min is None:
                    clip_min = file_clip_min
                    clip_max = file_clip_max
                else:
                    assert np.allclose(clip_min, file_clip_min), filename
                    assert np.allclose(clip_max, file_clip_max), filename

                assert episode_len >= 2 * horizon + 1, filename
                center_time_idxes = range(horizon, episode_len - horizon, horizon)
                condition_seq = np.stack(
                    [
                        build_condition(
                            rmb_data,
                            self.args.state_keys,
                            self.args.action_keys,
                            center_time_idx,
                            horizon,
                        )
                        for center_time_idx in center_time_idxes
                    ]
                )
                wrench_seq = np.concatenate(
                    [
                        build_wrench_target(rmb_data, center_time_idx, horizon)
                        for center_time_idx in center_time_idxes
                    ],
                    axis=0,
                )
                all_state.append(condition_seq)
                all_wrench.append(wrench_seq)

                if rgb_image_example is None:
                    rgb_image_example = {
                        camera_name: rmb_data[DataKey.get_rgb_image_key(camera_name)][0]
                        for camera_name in self.args.camera_names
                        if DataKey.get_rgb_image_key(camera_name) in rmb_data
                    }
                if depth_image_example is None:
                    depth_image_example = {
                        camera_name: rmb_data[DataKey.get_depth_image_key(camera_name)][
                            0
                        ]
                        for camera_name in self.args.camera_names
                        if DataKey.get_depth_image_key(camera_name) in rmb_data
                    }

        all_state = np.concatenate(all_state, dtype=np.float64)
        all_wrench = np.concatenate(all_wrench, dtype=np.float64)

        self.model_meta_info["state"].update(self.calc_stats_from_seq(all_state))
        self.model_meta_info["action"].update(self.calc_stats_from_seq(all_wrench))
        self.model_meta_info["action"]["percentile_clip"] = {
            "key": DataKey.MEASURED_EEF_WRENCH_MOVING_AVERAGE_PERCENTILE_CLIP,
            "source_key": DataKey.MEASURED_EEF_WRENCH_MOVING_AVERAGE,
            "min": clip_min,
            "max": clip_max,
        }
        self.model_meta_info["image"].update(
            {
                "rgb_example": rgb_image_example,
                "depth_example": depth_image_example,
            }
        )
        self.model_meta_info["data"].update(
            {
                "mean_episode_len": np.mean(episode_len_list),
                "min_episode_len": np.min(episode_len_list),
                "max_episode_len": np.max(episode_len_list),
            }
        )

    def setup_policy(self):
        first_camera_name = self.args.camera_names[0]
        image_height, image_width = self.model_meta_info["image"]["rgb_example"][
            first_camera_name
        ].shape[:2]

        # Set policy args
        self.model_meta_info["policy"]["args"] = {
            "lr": self.args.lr,
            "lr_material_property": self.args.lr_material_property,
            "weight_decay": 1e-4,
            "horizon": self.model_meta_info["data"]["horizon"],
            "hidden_dim": self.args.hidden_dim,
            "dim_feedforward": self.args.dim_feedforward,
            "lr_backbone": self.args.lr_backbone,
            "enc_layers": 4,
            "nheads": 8,
            "camera_names": self.args.camera_names,
            "image_shape": (image_height, image_width),
        }

        # Construct policy
        self.policy = WrenchPredictorPolicy(
            state_dim=len(self.model_meta_info["state"]["example"]),
            wrench_dim=len(self.model_meta_info["action"]["example"]),
            num_material_objects=len(
                self.model_meta_info["material_property"]["object_key_to_id"]
            ),
            material_property_dim=self.model_meta_info["material_property"]["dim"],
            policy_args=self.model_meta_info["policy"]["args"],
        )
        self.policy.cuda()

        self.freeze_zero_lr_parameter_groups()

        # Construct optimizer
        self.optimizer = self.policy.configure_optimizers()

        # Print policy information
        self.print_policy_info()
        print(f"  - horizon: {self.model_meta_info['data']['horizon']}")
        print(f"  - image size: {self.model_meta_info['data']['image_size']}")
        print(
            f"  - material objects: {self.model_meta_info['material_property']['object_key_to_id']}"
        )
        print(
            "  - lr material/model/backbone: "
            f"{self.args.lr_material_property}, {self.args.lr}, {self.args.lr_backbone}"
        )

    def freeze_zero_lr_parameter_groups(self):
        lr_by_group = {
            "material_property": self.args.lr_material_property,
            "backbone": self.args.lr_backbone,
            "model": self.args.lr,
        }
        frozen_param_count = {group: 0 for group in lr_by_group}

        for name, param in self.policy.named_parameters():
            group = self.get_parameter_group(name)
            if lr_by_group[group] <= 0.0:
                param.requires_grad = False
                frozen_param_count[group] += param.numel()

        trainable_param_count = sum(
            param.numel() for param in self.policy.parameters() if param.requires_grad
        )
        assert trainable_param_count > 0, "All WrenchPredictor3 parameters are frozen."
        print(
            f"[{self.__class__.__name__}] Frozen parameters: "
            f"material_property={frozen_param_count['material_property']}, "
            f"model={frozen_param_count['model']}, "
            f"backbone={frozen_param_count['backbone']}"
        )

    def get_parameter_group(self, name):
        if name.startswith("material_property_embedding"):
            return "material_property"
        if name.startswith("model.cnn"):
            return "backbone"
        return "model"

    def print_material_property_embedding(self, epoch):
        object_key_to_id = self.model_meta_info["material_property"]["object_key_to_id"]
        material_property = (
            self.policy.material_property_embedding.weight.detach().cpu().numpy()
        )
        print(f"[epoch {epoch}] material_property_embedding:")
        for object_key, object_id in sorted(
            object_key_to_id.items(), key=lambda item: item[1]
        ):
            print(f"  - {object_key}: {material_property[object_id]}")

    def train_loop(self):
        for epoch in tqdm(range(self.args.num_epochs)):
            # Run train step
            self.policy.train()
            batch_result_list = []
            for data in self.train_dataloader:
                self.optimizer.zero_grad()
                batch_result = self.policy(*[d.cuda() for d in data])
                loss = batch_result["loss"]
                loss.backward()
                self.optimizer.step()
                batch_result_list.append(self.detach_batch_result(batch_result))
            self.log_epoch_summary(batch_result_list, "train", epoch)

            # Run validation step
            with torch.inference_mode():
                self.policy.eval()
                batch_result_list = []
                for data in self.val_dataloader:
                    batch_result = self.policy(*[d.cuda() for d in data])
                    batch_result_list.append(self.detach_batch_result(batch_result))
                epoch_summary = self.log_epoch_summary(batch_result_list, "val", epoch)

                # Update best checkpoint
                self.update_best_ckpt(epoch_summary)

            # Save current checkpoint
            if epoch % max(self.args.num_epochs // 10, 1) == 0:
                self.print_material_property_embedding(epoch)
                self.save_current_ckpt(f"epoch{epoch:0>3}")

        # Save last checkpoint
        self.print_material_property_embedding("last")
        self.save_current_ckpt("last")

        # Save best checkpoint
        self.save_best_ckpt()
