import numpy as np
import torch

from robo_manip_baselines.common import (
    DataKey,
    DatasetBase,
    RmbData,
    get_skipped_data_seq,
    get_skipped_single_data,
)
from robo_manip_baselines.policy.wrench_predictor2.MaterialPropertyUtils import (
    get_material_object_id,
)


class WrenchPredictorDataset(DatasetBase):
    """Dataset to train WrenchPredictor3."""

    def setup_variables(self):
        skip = self.model_meta_info["data"]["skip"]
        self.sample_index_list = []
        for episode_idx, filename in enumerate(self.filenames):
            with RmbData(filename, self.enable_rmb_cache) as rmb_data:
                episode_len = rmb_data[DataKey.TIME][::skip].shape[0]
                assert episode_len >= 2
            for start_time_idx in range(episode_len - 1):
                self.sample_index_list.append((episode_idx, start_time_idx))

    def __len__(self):
        return len(self.sample_index_list)

    def __getitem__(self, sample_idx):
        skip = self.model_meta_info["data"]["skip"]
        chunk_size = self.model_meta_info["data"]["chunk_size"]
        episode_idx, start_time_idx = self.sample_index_list[sample_idx]
        filename = self.filenames[episode_idx]
        material_object_id = get_material_object_id(
            filename,
            self.model_meta_info["material_property"]["object_key_to_id"],
        )

        with RmbData(filename, self.enable_rmb_cache) as rmb_data:
            if len(self.model_meta_info["state"]["keys"]) == 0:
                state = np.zeros(0, dtype=np.float64)
            else:
                state_list = [
                    get_skipped_single_data(
                        rmb_data[key],
                        start_time_idx * skip,
                        key,
                        skip,
                    )
                    for key in self.model_meta_info["state"]["keys"]
                ]

                action_list = [
                    get_skipped_single_data(
                        rmb_data[key],
                        start_time_idx * skip,
                        key,
                        skip,
                    )
                    for key in self.model_meta_info["action"]["keys"]
                ]

                state = np.concatenate(state_list + action_list)

            wrench = get_skipped_data_seq(
                rmb_data[DataKey.MEASURED_EEF_WRENCH_MOVING_AVERAGE_PERCENTILE_CLIP][
                    (start_time_idx + 1) * skip :
                ],
                DataKey.MEASURED_EEF_WRENCH_MOVING_AVERAGE_PERCENTILE_CLIP,
                skip,
            )

        wrench_len = min(wrench.shape[0], chunk_size)
        wrench_chunked = np.zeros((chunk_size, wrench.shape[1]), dtype=np.float64)
        wrench_chunked[:wrench_len] = wrench[:wrench_len]
        is_pad = np.zeros(chunk_size, dtype=bool)
        is_pad[wrench_len:] = True

        state, wrench_chunked, _ = self.pre_convert_data(state, wrench_chunked, None)

        state_tensor = torch.tensor(state, dtype=torch.float32)
        wrench_tensor = torch.tensor(wrench_chunked, dtype=torch.float32)
        is_pad_tensor = torch.tensor(is_pad, dtype=torch.bool)
        material_object_id_tensor = torch.tensor(material_object_id, dtype=torch.long)

        state_tensor, wrench_tensor, _ = self.augment_data(
            state_tensor,
            wrench_tensor,
            None,
        )

        return (
            state_tensor,
            material_object_id_tensor,
            wrench_tensor,
            is_pad_tensor,
        )
