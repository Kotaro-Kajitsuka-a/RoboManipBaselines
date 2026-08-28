import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from robo_manip_baselines.common import (
    DataKey,
    RmbData,
    convert_data_to_policy,
    find_rmb_files,
)
from robo_manip_baselines.misc.futureimagination.ImageJointConditionedCVAE import (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    load_image_joint_conditioned_cvae,
)

BATCH_SIZE = 64


def parse_args():
    parser = argparse.ArgumentParser(
        description="Add deterministic joint-conditioned CVAE features to RMB data."
    )
    parser.add_argument("path", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--camera_name", required=True)
    parser.add_argument("--output_key", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_condition(rmb_data, condition_keys):
    return np.concatenate(
        [convert_data_to_policy(rmb_data[key][:], key) for key in condition_keys],
        axis=1,
    ).astype(np.float32)


def encode_episode(model, rmb_path, rgb_image_key):
    with RmbData(
        rmb_path,
        image_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    ) as rmb_data:
        video = rmb_data[rgb_image_key][:]
        condition = load_condition(rmb_data, model.condition_keys)
    assert len(video) == len(condition), (rmb_path, len(video), len(condition))

    features = []
    for start in range(0, len(video), BATCH_SIZE):
        images = torch.from_numpy(video[start : start + BATCH_SIZE])
        images = images.cuda().permute(0, 3, 1, 2).float() / 255.0
        condition_batch = torch.from_numpy(condition[start : start + BATCH_SIZE]).cuda()
        with torch.inference_mode():
            posterior_mean, _log_variance = model.encode(images, condition_batch)
        features.append(posterior_mean.cpu())
    return torch.cat(features).numpy()


def main():
    args = parse_args()
    assert args.checkpoint.is_dir(), args.checkpoint
    assert torch.cuda.is_available(), "Image CVAE encoding requires a CUDA GPU."
    model = load_image_joint_conditioned_cvae(args.checkpoint, device="cuda").eval()
    model.requires_grad_(False)
    rgb_image_key = DataKey.get_rgb_image_key(args.camera_name)

    rmb_paths = find_rmb_files(str(args.path))
    assert rmb_paths, args.path
    for rmb_path in tqdm(rmb_paths, unit="episode"):
        features = encode_episode(model, rmb_path, rgb_image_key)
        assert features.shape[1] == model.latent_dim, features.shape
        with RmbData(rmb_path, mode="r+") as rmb_data:
            if args.output_key in rmb_data:
                if not args.overwrite:
                    raise FileExistsError(f"{rmb_path}: {args.output_key}")
                del rmb_data.h5file[args.output_key]
            dataset = rmb_data.h5file.create_dataset(args.output_key, data=features)
            dataset.attrs["model"] = str(args.checkpoint)
            dataset.attrs["source_camera"] = args.camera_name
            dataset.attrs["source_image_key"] = rgb_image_key
            dataset.attrs["condition_keys"] = json.dumps(model.condition_keys)
            dataset.attrs["condition_dim"] = model.condition_dim
            dataset.attrs["latent_value"] = "posterior_mean"
            dataset.attrs["latent_dim"] = model.latent_dim

    print(f"HDF5 key: {args.output_key}")
    print(f"Feature shape per episode: (T, {model.latent_dim})")
    print(f"Episodes: {len(rmb_paths)}")


if __name__ == "__main__":
    main()
