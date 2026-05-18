import numpy as np

from robo_manip_baselines.common import DataKey, get_skipped_single_data


WRENCH_TARGET_KEY = DataKey.MEASURED_EEF_WRENCH_MOVING_AVERAGE_PERCENTILE_CLIP


def get_raw_single_data(rmb_data, key, time_idx):
    return get_skipped_single_data(rmb_data[key], time_idx, key, skip=1)


def build_data_vector(rmb_data, keys, time_idx):
    if len(keys) == 0:
        return np.zeros(0, dtype=np.float64)

    return np.concatenate(
        [get_raw_single_data(rmb_data, key, time_idx) for key in keys]
    )


def build_condition(rmb_data, state_keys, action_keys, center_time_idx, horizon):
    action_time_idxes = range(center_time_idx, center_time_idx + horizon)

    state = build_data_vector(rmb_data, state_keys, center_time_idx)
    action = np.concatenate(
        [
            build_data_vector(rmb_data, action_keys, time_idx)
            for time_idx in action_time_idxes
        ]
    )
    return np.concatenate([state, action])


def build_wrench_target(rmb_data, center_time_idx, horizon):
    return np.asarray(
        rmb_data[WRENCH_TARGET_KEY][
            center_time_idx + 1 : center_time_idx + horizon + 1
        ],
        dtype=np.float64,
    )
