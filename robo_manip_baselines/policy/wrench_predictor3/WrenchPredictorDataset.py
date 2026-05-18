import numpy as np
import torch

from robo_manip_baselines.common import (
    DataKey,
    DatasetBase,
    RmbData,
)

from .MaterialPropertyUtils import get_material_object_id
from .WrenchPredictorSequenceUtils import build_condition, build_wrench_target


class WrenchPredictorDataset(DatasetBase):
    """Dataset to train WrenchPredictor3."""

    IMAGE_HISTORY_SIZE = 2

    def setup_variables(self):
        skip = self.model_meta_info["data"]["skip"]
        self.sample_index_list = []
        for episode_idx, filename in enumerate(self.filenames):
            with RmbData(filename, self.enable_rmb_cache) as rmb_data:
                episode_len = rmb_data[DataKey.TIME][::skip].shape[0]
                assert episode_len >= self.IMAGE_HISTORY_SIZE + 1
            for start_time_idx in range(self.IMAGE_HISTORY_SIZE - 1, episode_len - 1):
                self.sample_index_list.append((episode_idx, start_time_idx))

    def __len__(self):
        return len(self.sample_index_list)

    def __getitem__(self, sample_idx):
        skip = self.model_meta_info["data"]["skip"]
        image_size = self.model_meta_info["data"]["image_size"]
        horizon = skip
        episode_idx, start_time_idx = self.sample_index_list[sample_idx]
        center_time_idx = start_time_idx * skip
        filename = self.filenames[episode_idx]
        material_object_id = get_material_object_id(
            filename,
            self.model_meta_info["material_property"]["object_key_to_id"],
        )

        with RmbData(
            filename, self.enable_rmb_cache, image_size=image_size
        ) as rmb_data:
            condition = build_condition(
                rmb_data,
                self.model_meta_info["state"]["keys"],
                self.model_meta_info["action"]["keys"],
                center_time_idx,
                horizon,
            )
            wrench = build_wrench_target(rmb_data, center_time_idx, horizon)

            # Load images
            image_time_idxes = np.array([start_time_idx - 1, start_time_idx])
            image_keys = [
                DataKey.get_rgb_image_key(camera_name)
                for camera_name in self.model_meta_info["image"]["camera_names"]
            ]
            images = np.stack(
                [
                    # This allows for a common hash of cache
                    rmb_data[key][::skip][image_time_idxes]
                    if self.enable_rmb_cache
                    # This allows for minimal loading when reading from HDF5
                    else np.stack(
                        [
                            rmb_data[key][int(time_idx * skip)]
                            for time_idx in image_time_idxes
                        ],
                        axis=0,
                    )
                    for key in image_keys
                ],
                axis=0,
            )

        # Pre-convert data
        condition, wrench, images = self.pre_convert_data(condition, wrench, images)

        # Convert to tensor
        condition_tensor = torch.tensor(condition, dtype=torch.float32)
        wrench_tensor = torch.tensor(wrench, dtype=torch.float32)
        images_tensor = torch.tensor(images, dtype=torch.uint8)
        material_object_id_tensor = torch.tensor(material_object_id, dtype=torch.long)

        # Augment data
        condition_tensor, wrench_tensor, images_tensor = self.augment_data(
            condition_tensor, wrench_tensor, images_tensor
        )

        # Sort in the order of policy inputs and outputs
        return (
            condition_tensor,
            images_tensor,
            material_object_id_tensor,
            wrench_tensor,
        )
