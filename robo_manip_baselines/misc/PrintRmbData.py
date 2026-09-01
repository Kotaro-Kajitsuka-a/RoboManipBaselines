import argparse

import numpy as np

from robo_manip_baselines.common import DataKey, RmbData, find_rmb_files


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
        "--only_stats",
        action="store_true",
        help="whether to print only statistics for the entire data set",
    )

    return parser.parse_args()


class PrintRmbData:
    def __init__(self, path, only_stats):
        self.path = path
        self.only_stats = only_stats

    def run(self):
        rmb_path_list = find_rmb_files(self.path)

        stats = {
            entry: []
            for entry in [
                "episode_len",
                "success_once",
                "success_last",
            ]
        }

        # 各 timestep 間の dt を保存する
        dt_list = []

        for rmb_path in rmb_path_list:
            print(f"[{self.__class__.__name__}] Open {rmb_path}")

            try:
                with RmbData(rmb_path) as rmb_data:
                    time = np.asarray(rmb_data[DataKey.TIME][:])

                    stats["episode_len"].append(len(time))

                    # timestep 間の dt を計算
                    # dt[i] = time[i + 1] - time[i]
                    if len(time) >= 2:
                        dt = np.diff(time)
                        dt_list.extend(dt)

                    if DataKey.REWARD in rmb_data.keys():
                        stats["success_once"].append(
                            np.any(rmb_data[DataKey.REWARD][:] > 0.0)
                        )
                        stats["success_last"].append(rmb_data[DataKey.REWARD][-1] > 0.0)
                    else:
                        stats["success_once"].append(False)
                        stats["success_last"].append(False)

                    if self.only_stats:
                        continue

                    print("  Attrs:")
                    for k, v in rmb_data.attrs.items():
                        print(f"    - {k}: {v}")

                    print("  Data:")
                    for k in rmb_data.keys():
                        v = rmb_data[k]
                        print(f"    - {k}: {v.shape}, [{v.dtype}]")

            except (OSError, IOError, ValueError) as e:
                print(f"[Error] Failed to load {rmb_path}: {e}")

        stats = {k: np.array(v) for k, v in stats.items()}
        dt_array = np.array(dt_list)

        print(f"[{self.__class__.__name__}] Statistics of the entire data set:")

        print(
            f"  - episode len mean: {int(stats['episode_len'].mean())}, "
            f"std: {int(stats['episode_len'].std())}, "
            f"min: {stats['episode_len'].min()}, "
            f"max: {stats['episode_len'].max()}"
        )

        print(
            f"  - success once: {np.sum(stats['success_once'])} / {len(rmb_path_list)},  "
            f"last: {np.sum(stats['success_last'])} / {len(rmb_path_list)}"
        )

        # timestep 間の dt の平均値と標準偏差を出力
        if len(dt_array) > 0:
            print(
                f"  - timestep dt mean: {dt_array.mean():.6f} s, "
                f"std: {dt_array.std():.6f} s"
            )
        else:
            print("  - timestep dt mean/std: N/A (not enough timesteps)")


if __name__ == "__main__":
    print_data = PrintRmbData(**vars(parse_argument()))
    print_data.run()
