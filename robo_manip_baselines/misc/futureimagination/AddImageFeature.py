import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from robo_manip_baselines.common import DataKey, RmbData, find_rmb_files
from robo_manip_baselines.misc.futureimagination.SD3_module import VAE as SD3VAE


MODEL_NAME = "stabilityai/stable-diffusion-3-medium-diffusers"
CAMERA_NAME = "left"
RGB_IMAGE_KEY = DataKey.get_rgb_image_key(CAMERA_NAME)
IMAGE_FEATURE_KEY = "sd3_vae"
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 96
LATENT_CHANNELS = 16
LATENT_HEIGHT = 12
LATENT_WIDTH = 16
IMAGE_FEATURE_DIM = LATENT_CHANNELS * LATENT_HEIGHT * LATENT_WIDTH
BATCH_SIZE = 16


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Add flattened SD3 VAE features from every left-camera frame to RMB HDF5 files."
        )
    )
    parser.add_argument(
        "path",
        type=Path,
        help="RMB episode or directory containing RMB episodes",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=f"replace an existing '{IMAGE_FEATURE_KEY}' dataset",
    )
    return parser.parse_args()


def preflight(rmb_paths, overwrite):
    total_num_frames = 0
    for rmb_path in rmb_paths:
        with RmbData(rmb_path) as rmb_data:
            assert RGB_IMAGE_KEY in rmb_data, (rmb_path, RGB_IMAGE_KEY)
            num_video_frames = len(rmb_data[RGB_IMAGE_KEY])

            if IMAGE_FEATURE_KEY in rmb_data and not overwrite:
                raise FileExistsError(
                    f"{rmb_path}: '{IMAGE_FEATURE_KEY}' already exists "
                    "(use --overwrite to replace it)"
                )

        total_num_frames += num_video_frames
    return total_num_frames


def encode_image_batch(sd3_vae, rgb_images, device):
    images = torch.from_numpy(np.stack(rgb_images)).to(
        device=device,
        dtype=torch.bfloat16,
    )
    images = images.permute(0, 3, 1, 2) / 127.5 - 1.0

    vae = sd3_vae.vae
    with torch.inference_mode():
        latents = vae.encode(images).latent_dist.mode()
        latents = (
            latents - vae.config.shift_factor
        ) * vae.config.scaling_factor  # soft normalization (-1~1) to latent space

    assert latents.shape[1:] == (
        LATENT_CHANNELS,
        LATENT_HEIGHT,
        LATENT_WIDTH,
    ), latents.shape

    features = latents.float().flatten(start_dim=1).cpu().numpy()
    assert features.shape == (len(rgb_images), IMAGE_FEATURE_DIM), features.shape
    return features.astype(np.float16)


def encode_episode(sd3_vae, rmb_path, device):
    with RmbData(rmb_path) as rmb_data:
        rgb_video = rmb_data[RGB_IMAGE_KEY][:]

    feature_batches = []
    for start in range(0, len(rgb_video), BATCH_SIZE):
        rgb_images = [
            cv2.resize(
                rgb_image,
                (IMAGE_WIDTH, IMAGE_HEIGHT),
                interpolation=cv2.INTER_LINEAR,
            )
            for rgb_image in rgb_video[start : start + BATCH_SIZE]
        ]
        feature_batches.append(encode_image_batch(sd3_vae, rgb_images, device))

    features = np.concatenate(feature_batches, axis=0)
    assert features.ndim == 2
    assert features.shape[1] == IMAGE_FEATURE_DIM
    return features


def write_features(rmb_path, features, sd3_vae, overwrite):
    with RmbData(rmb_path, mode="r+") as rmb_data:
        h5file = rmb_data.h5file

        if IMAGE_FEATURE_KEY in h5file:
            assert overwrite
            del h5file[IMAGE_FEATURE_KEY]

        dataset = h5file.create_dataset(
            IMAGE_FEATURE_KEY,
            data=features,
            chunks=(min(BATCH_SIZE, features.shape[0]), IMAGE_FEATURE_DIM),
            compression="lzf",
            shuffle=True,
        )
        dataset.attrs["model"] = MODEL_NAME
        dataset.attrs["source_camera"] = CAMERA_NAME
        dataset.attrs["source_image_key"] = RGB_IMAGE_KEY
        dataset.attrs["resized_image_size"] = [IMAGE_WIDTH, IMAGE_HEIGHT]
        dataset.attrs["latent_shape_chw"] = [
            LATENT_CHANNELS,
            LATENT_HEIGHT,
            LATENT_WIDTH,
        ]
        dataset.attrs["flatten_order"] = "CHW"
        dataset.attrs["latent_distribution_value"] = "mode"
        dataset.attrs["scaling_factor"] = sd3_vae.vae.config.scaling_factor
        dataset.attrs["shift_factor"] = sd3_vae.vae.config.shift_factor


def main():
    args = parse_args()
    rmb_paths = find_rmb_files(str(args.path))
    total_num_frames = preflight(rmb_paths, args.overwrite)

    assert torch.cuda.is_available(), "This operation requires a CUDA GPU."
    device = torch.device("cuda")
    sd3_vae = SD3VAE().to(device)

    progress = tqdm(rmb_paths, unit="episode")
    processed_num_frames = 0
    for rmb_path in progress:
        progress.set_description(Path(rmb_path).name)
        features = encode_episode(sd3_vae, rmb_path, device)
        write_features(rmb_path, features, sd3_vae, args.overwrite)
        processed_num_frames += features.shape[0]

    assert processed_num_frames == total_num_frames
    print(f"HDF5 key: {IMAGE_FEATURE_KEY}")
    print(f"Feature shape per episode: (T, {IMAGE_FEATURE_DIM})")
    print(f"Episodes: {len(rmb_paths)}")
    print(f"Frames: {processed_num_frames}")


if __name__ == "__main__":
    main()
