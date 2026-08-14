import argparse
from pathlib import Path

import torch
from pythae.models import AutoModel
from tqdm import tqdm

from robo_manip_baselines.common import DataKey, RmbData, find_rmb_files


BATCH_SIZE = 64


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--camera_name", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def encode_episode(model, rmb_path, rgb_image_key, image_size):
    with RmbData(rmb_path, image_size=image_size) as rmb_data:
        video = rmb_data[rgb_image_key][:]

    features = []
    for start in range(0, len(video), BATCH_SIZE):
        images = torch.from_numpy(video[start : start + BATCH_SIZE])
        images = images.cuda().permute(0, 3, 1, 2).float() / 255.0
        with torch.inference_mode():
            features.append(model.encoder(images).embedding.cpu())
    return torch.cat(features).numpy()


def main():
    args = parse_args()
    assert torch.cuda.is_available(), "Image VAE encoding requires a CUDA GPU."
    model = AutoModel.load_from_folder(str(args.checkpoint)).eval().cuda()

    _, image_height, image_width = model.model_config.input_dim
    latent_dim = model.model_config.latent_dim
    rgb_image_key = DataKey.get_rgb_image_key(args.camera_name)
    image_feature_key = f"image_vae_{args.camera_name}_{latent_dim}"

    for rmb_path in tqdm(find_rmb_files(str(args.path)), unit="episode"):
        features = encode_episode(
            model,
            rmb_path,
            rgb_image_key,
            (image_width, image_height),
        )
        assert features.shape[1] == latent_dim, features.shape
        with RmbData(rmb_path, mode="r+") as rmb_data:
            if image_feature_key in rmb_data:
                if not args.overwrite:
                    raise FileExistsError(f"{rmb_path}: {image_feature_key}")
                del rmb_data.h5file[image_feature_key]
            dataset = rmb_data.h5file.create_dataset(image_feature_key, data=features)
            dataset.attrs["model"] = str(args.checkpoint)
            dataset.attrs["source_camera"] = args.camera_name
            dataset.attrs["source_image_key"] = rgb_image_key
            dataset.attrs["latent_dim"] = latent_dim


if __name__ == "__main__":
    main()
