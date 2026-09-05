import numpy as np
import torch

from robo_manip_baselines.common import (
    DataKey,
    DatasetBase,
    RmbData,
    convert_data_to_policy,
    get_skipped_data_seq,
    normalize_data,
)


def _load_wrench(rmb_data, model_meta_info):
    wrench_info = model_meta_info["wrench"]
    wrench_key = wrench_info["key"]
    if wrench_key in rmb_data.keys():
        return np.asarray(rmb_data[wrench_key][:])

    if "percentile_clip" not in wrench_info:
        raise KeyError(
            f"'{wrench_key}' is not found and the checkpoint does not contain "
            "percentile clipping metadata"
        )

    clip_info = wrench_info["percentile_clip"]
    source_key = clip_info["source_key"]
    if source_key not in rmb_data.keys():
        raise KeyError(f"Neither '{wrench_key}' nor its source '{source_key}' is found")

    source_wrench = np.asarray(rmb_data[source_key][:])
    return np.clip(
        source_wrench,
        clip_info["min"],
        clip_info["max"],
    ).astype(source_wrench.dtype, copy=False)


class WrenchPredictor4OnlineDataset(DatasetBase):
    """One episode split into chronological, fully observed prediction windows."""

    def setup_variables(self):
        assert len(self.filenames) == 1, self.filenames

        skip = self.model_meta_info["data"]["skip"]
        horizon = self.model_meta_info["data"]["horizon"]
        with RmbData(self.filenames[0]) as rmb_data:
            episode_len = rmb_data[DataKey.TIME][::skip].shape[0]

        assert episode_len >= horizon, (episode_len, horizon)
        self.start_time_idxes = list(range(episode_len - horizon + 1))

    def __len__(self):
        return len(self.start_time_idxes)

    def __getitem__(self, chunk_idx):
        skip = self.model_meta_info["data"]["skip"]
        horizon = self.model_meta_info["data"]["horizon"]
        start_time_idx = self.start_time_idxes[chunk_idx]
        time_idxes = np.arange(start_time_idx, start_time_idx + horizon)

        with RmbData(self.filenames[0], self.enable_rmb_cache) as rmb_data:
            time = rmb_data[DataKey.TIME][::skip]

            image_feature_key = self.model_meta_info["data"]["image_feature_key"]
            image_feature = convert_data_to_policy(
                get_skipped_data_seq(
                    rmb_data[image_feature_key][:],
                    image_feature_key,
                    skip,
                )[time_idxes],
                image_feature_key,
            )

            state_keys = self.model_meta_info["state"]["keys"]
            if len(state_keys) == 0:
                state = np.zeros((horizon, 0), dtype=np.float64)
            else:
                state = np.concatenate(
                    [
                        convert_data_to_policy(
                            get_skipped_data_seq(rmb_data[key][:], key, skip)[
                                time_idxes
                            ],
                            key,
                        )
                        for key in state_keys
                    ],
                    axis=1,
                )

            action = np.concatenate(
                [
                    convert_data_to_policy(
                        get_skipped_data_seq(rmb_data[key][:], key, skip)[time_idxes],
                        key,
                    )
                    for key in self.model_meta_info["action"]["keys"]
                ],
                axis=1,
            )

            wrench_key = self.model_meta_info["wrench"]["key"]
            wrench = get_skipped_data_seq(
                _load_wrench(rmb_data, self.model_meta_info),
                wrench_key,
                skip,
            )[time_idxes]

        state, action, _ = super().pre_convert_data(state, action, None)
        image_feature = normalize_data(
            image_feature,
            self.model_meta_info["image_feature"],
        )
        wrench = normalize_data(wrench, self.model_meta_info["wrench"])

        return {
            "image_feature": torch.tensor(image_feature, dtype=torch.float32),
            "state": torch.tensor(state, dtype=torch.float32),
            "action": torch.tensor(action, dtype=torch.float32),
            "wrench": torch.tensor(wrench, dtype=torch.float32),
            "observed_time": torch.tensor(
                time[time_idxes[-1]] - time[0],
                dtype=torch.float32,
            ),
        }
