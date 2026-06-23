import numpy as np
import torch

from robo_manip_baselines.common import (
    DataKey,
    DatasetBase,
    DpStyleDatasetMixin,
    RmbData,
    get_skipped_data_seq,
    normalize_data,
)


class DiffusionWorldModelDataset(DatasetBase, DpStyleDatasetMixin):
    """Dataset to train diffusion world model."""

    IMAGE_FEATURE_KEY = "diffusion_policy_obs_visual_feature"
    WRENCH_KEY = DataKey.MEASURED_EEF_WRENCH_MOVING_AVERAGE_PERCENTILE_CLIP
    OBJECT_KEY_TO_ID = {
        "WrenchPredObject0": 0,
        "WrenchPredObject1": 1,
        "WrenchPredObject2": 2,
        "WrenchPredObject3": 3,
    }

    def setup_variables(self):
        self.setup_dp_style_chunk()

    def __len__(self):
        return len(self.chunk_info_list)

    def __getitem__(self, chunk_idx):
        skip = self.model_meta_info["data"]["skip"]
        horizon = self.model_meta_info["data"]["horizon"]
        episode_idx, start_time_idx = self.chunk_info_list[chunk_idx]
        filename = self.filenames[episode_idx]
        object_id = self.get_object_id(filename)

        with RmbData(filename, self.enable_rmb_cache) as rmb_data:
            episode_len = rmb_data[DataKey.TIME][::skip].shape[0]
            time_idxes = np.clip(
                np.arange(start_time_idx, start_time_idx + horizon), 0, episode_len - 1
            )

            # Load image feature
            image_feature = get_skipped_data_seq(
                rmb_data[self.IMAGE_FEATURE_KEY][:], self.IMAGE_FEATURE_KEY, skip
            )[time_idxes]

            if len(self.model_meta_info["state"]["keys"]) == 0:
                state = np.zeros((horizon, 0), dtype=np.float64)
            else:
                state = np.concatenate(
                    [
                        get_skipped_data_seq(rmb_data[key][:], key, skip)[time_idxes]
                        for key in self.model_meta_info["state"]["keys"]
                    ],
                    axis=1,
                )

            # Load action
            action = np.concatenate(
                [
                    get_skipped_data_seq(rmb_data[key][:], key, skip)[time_idxes]
                    for key in self.model_meta_info["action"]["keys"]
                ],
                axis=1,
            )

            wrench = get_skipped_data_seq(
                rmb_data[self.WRENCH_KEY][:], self.WRENCH_KEY, skip
            )[time_idxes]

        state, action, image_feature, wrench = self.pre_convert_data(
            state, action, image_feature, wrench
        )

        return {
            "image_feature": torch.tensor(image_feature, dtype=torch.float32),
            "state": torch.tensor(state, dtype=torch.float32),
            "action": torch.tensor(action, dtype=torch.float32),
            "wrench": torch.tensor(wrench, dtype=torch.float32),
            "object_id": torch.tensor(object_id, dtype=torch.long),
        }

    def pre_convert_data(self, state, action, image_feature, wrench):
        state, action, _images = super().pre_convert_data(state, action, None)
        image_feature = normalize_data(
            image_feature, self.model_meta_info["image_feature"]
        )
        wrench = normalize_data(wrench, self.model_meta_info["wrench"])
        return state, action, image_feature, wrench

    def get_object_id(self, filename):
        matched_object_ids = [
            object_id
            for object_key, object_id in self.OBJECT_KEY_TO_ID.items()
            if object_key in filename
        ]
        assert len(matched_object_ids) == 1, (filename, matched_object_ids)
        return matched_object_ids[0]
