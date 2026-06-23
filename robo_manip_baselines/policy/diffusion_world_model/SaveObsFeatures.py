import argparse

import numpy as np
import torch
from torchvision.transforms import v2
from tqdm import tqdm

from robo_manip_baselines.common import (
    DataKey,
    RmbData,
    find_rmb_files,
    normalize_data,
)

from robo_manip_baselines.policy.diffusion_world_model.DiffusionPolicyObsEncoder import (
    FrozenDiffusionPolicyObsEncoder,
)


def parse_argument():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "checkpoint",
        type=str,
        help="trained Diffusion Policy checkpoint",
    )
    parser.add_argument(
        "path",
        type=str,
        help="path to data (*.hdf5 or *.rmb) or directory containing them",
    )
    parser.add_argument(
        "--image_feature_key",
        type=str,
        default="diffusion_policy_obs_visual_feature",
        help="key to save frozen visual features",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="batch size for feature extraction",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="whether to overwrite existing value if it exists",
    )
    return parser.parse_args()


class SaveObsFeatures:
    def __init__(
        self,
        checkpoint,
        path,
        image_feature_key="diffusion_policy_obs_visual_feature",
        batch_size=128,
        overwrite=False,
    ):
        self.checkpoint = checkpoint
        self.path = path
        self.image_feature_key = image_feature_key
        self.batch_size = batch_size
        self.device = torch.device("cuda")
        self.overwrite = overwrite

        self.encoder = FrozenDiffusionPolicyObsEncoder(checkpoint, device="cuda")
        self.model_meta_info = self.encoder.model_meta_info
        self.image_transforms = v2.Compose([v2.ToDtype(torch.float32, scale=True)])

    def run(self):
        print(f"[{self.__class__.__name__}] Load checkpoint: {self.checkpoint}")
        print(
            f"[{self.__class__.__name__}] Save visual feature to '{self.image_feature_key}'."
        )
        print(f"  - obs feature shape: {self.encoder.output_shape()}")
        print(f"  - visual feature shape: {self.encoder.visual_output_shape()}")
        print(f"  - feature slices: {self.encoder.feature_slices}")

        rmb_path_list = find_rmb_files(self.path)
        for rmb_path in tqdm(rmb_path_list):
            tqdm.write(f"[{self.__class__.__name__}] Open {rmb_path}")
            with RmbData(
                rmb_path,
                mode="r+",
                image_size=self.model_meta_info["data"]["image_size"],
            ) as rmb_data:
                if self.image_feature_key in rmb_data.keys():
                    if self.overwrite:
                        del rmb_data.h5file[self.image_feature_key]
                    else:
                        raise ValueError(
                            f"[{self.__class__.__name__}] '{self.image_feature_key}' already exists: "
                            f"{rmb_path} (use --overwrite to replace)"
                        )

                feature = self.extract_feature(rmb_data)
                rmb_data.h5file.create_dataset(self.image_feature_key, data=feature)
                rmb_data.attrs[self.image_feature_key + "_checkpoint"] = (
                    self.checkpoint
                )
                rmb_data.attrs[self.image_feature_key + "_feature_type"] = "visual"
                rmb_data.attrs[self.image_feature_key + "_output_shape"] = (
                    feature.shape[1:]
                )
                rmb_data.attrs[self.image_feature_key + "_camera_names"] = np.array(
                    self.model_meta_info["image"]["camera_names"], dtype="S"
                )
                rmb_data.attrs[self.image_feature_key + "_state_keys"] = np.array(
                    self.model_meta_info["state"]["keys"], dtype="S"
                )

    def extract_feature(self, rmb_data):
        time_len = rmb_data[DataKey.TIME].shape[0]
        feature_list = []
        for start in range(0, time_len, self.batch_size):
            stop = min(start + self.batch_size, time_len)
            obs_dict = self.make_obs_dict(rmb_data, slice(start, stop))
            feature = self.encoder.encode_visual(obs_dict)
            feature_list.append(feature.cpu().numpy())
        return np.concatenate(feature_list, axis=0)

    def make_obs_dict(self, rmb_data, time_slice):
        obs_dict = {}

        if len(self.model_meta_info["state"]["keys"]) > 0:
            state = np.concatenate(
                [
                    np.asarray(rmb_data[key][time_slice])
                    for key in self.model_meta_info["state"]["keys"]
                ],
                axis=1,
            )
            state = normalize_data(state, self.model_meta_info["state"])
            obs_dict["state"] = torch.tensor(
                state, dtype=torch.float32, device=self.device
            )

        for camera_name in self.model_meta_info["image"]["camera_names"]:
            image_key = DataKey.get_rgb_image_key(camera_name)
            image = np.asarray(rmb_data[image_key][time_slice])
            image = np.moveaxis(image, -1, -3)
            image = torch.tensor(image, dtype=torch.uint8)
            image = self.image_transforms(image)
            image = image * 2.0 - 1.0
            obs_dict[image_key] = image.to(self.device)

        return obs_dict


if __name__ == "__main__":
    save_obs_features = SaveObsFeatures(**vars(parse_argument()))
    save_obs_features.run()
