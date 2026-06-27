import argparse

import numpy as np
from tqdm import tqdm

from robo_manip_baselines.common import DataKey, RmbData, find_rmb_files


def get_percentile_clip_wrench_key(src_key):
    if src_key == DataKey.MEASURED_EEF_WRENCH:
        return "measured_eef_wrench_percentile_clip"
    elif src_key == DataKey.MEASURED_EEF_WRENCH_MOVING_AVERAGE:
        return DataKey.MEASURED_EEF_WRENCH_MOVING_AVERAGE_PERCENTILE_CLIP
    else:
        raise ValueError(
            f"[get_percentile_clip_wrench_key] Invalid source key: {src_key}"
        )


def parse_argument():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "path",
        type=str,
        help="path to data (*.hdf5 or *.rmb) or directory containing them",
    )
    parser.add_argument(
        "--low_percentile",
        type=float,
        default=1.0,
        help="lower percentile used for clipping",
    )
    parser.add_argument(
        "--high_percentile",
        type=float,
        default=99.0,
        help="upper percentile used for clipping",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="whether to overwrite existing value if it exists",
    )
    parser.add_argument(
        "--src_key",
        type=str,
        default=DataKey.MEASURED_EEF_WRENCH_MOVING_AVERAGE,
        choices=[
            DataKey.MEASURED_EEF_WRENCH,
            DataKey.MEASURED_EEF_WRENCH_MOVING_AVERAGE,
        ],
        help="source wrench key to percentile-clip",
    )
    parser.add_argument(
        "--dst_key",
        type=str,
        default=None,
        help="destination key for percentile-clipped wrench",
    )

    return parser.parse_args()


class AddPercentileClippedWrenchToRmbData:
    def __init__(
        self,
        path,
        low_percentile=1.0,
        high_percentile=99.0,
        overwrite=False,
        src_key=DataKey.MEASURED_EEF_WRENCH_MOVING_AVERAGE,
        dst_key=None,
    ):
        self.path = path
        self.low_percentile = low_percentile
        self.high_percentile = high_percentile
        self.overwrite = overwrite
        self.src_key = src_key
        self.dst_key = dst_key or get_percentile_clip_wrench_key(src_key)

    def run(self):
        if not self.low_percentile < self.high_percentile:
            raise ValueError(
                f"[{self.__class__.__name__}] low_percentile must be smaller than high_percentile: "
                f"{self.low_percentile}, {self.high_percentile}"
            )

        rmb_path_list = find_rmb_files(self.path)
        clip_min, clip_max = self.calc_clip_bounds(rmb_path_list)
        print(
            f"[{self.__class__.__name__}] Add '{self.dst_key}' clipped from '{self.src_key}'."
        )
        print(f"  - low percentile: {self.low_percentile}, values: {clip_min}")
        print(f"  - high percentile: {self.high_percentile}, values: {clip_max}")

        for rmb_path in tqdm(rmb_path_list):
            tqdm.write(f"[{self.__class__.__name__}] Open {rmb_path}")
            with RmbData(rmb_path, mode="r+") as rmb_data:
                if self.dst_key in rmb_data.keys():
                    if self.overwrite:
                        del rmb_data.h5file[self.dst_key]
                    else:
                        raise ValueError(
                            f"[{self.__class__.__name__}] '{self.dst_key}' already exists: "
                            f"{rmb_path} (use --overwrite to replace)"
                        )

                wrench = np.asarray(rmb_data[self.src_key][:])
                clipped_wrench = np.clip(wrench, clip_min, clip_max).astype(
                    wrench.dtype,
                    copy=False,
                )
                rmb_data.h5file[self.dst_key] = clipped_wrench
                rmb_data.attrs[self.dst_key + "_source_key"] = self.src_key
                rmb_data.attrs[self.dst_key + "_low_percentile"] = self.low_percentile
                rmb_data.attrs[self.dst_key + "_high_percentile"] = self.high_percentile
                rmb_data.attrs[self.dst_key + "_clip_min"] = clip_min
                rmb_data.attrs[self.dst_key + "_clip_max"] = clip_max

    def calc_clip_bounds(self, rmb_path_list):
        wrench_list = []
        for rmb_path in rmb_path_list:
            with RmbData(rmb_path) as rmb_data:
                if self.src_key not in rmb_data.keys():
                    raise KeyError(
                        f"[{self.__class__.__name__}] '{self.src_key}' is not found: {rmb_path}"
                    )
                wrench_list.append(
                    np.asarray(rmb_data[self.src_key][:], dtype=np.float64)
                )

        wrench = np.concatenate(wrench_list, axis=0)
        clip_min = np.percentile(wrench, self.low_percentile, axis=0)
        clip_max = np.percentile(wrench, self.high_percentile, axis=0)
        return clip_min, clip_max


if __name__ == "__main__":
    add_clipped_wrench = AddPercentileClippedWrenchToRmbData(**vars(parse_argument()))
    add_clipped_wrench.run()
