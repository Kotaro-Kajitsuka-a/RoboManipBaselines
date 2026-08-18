import argparse
from pathlib import Path

import numpy as np
import torch
from pythae.models import AutoModel
from tqdm import tqdm

from robo_manip_baselines.common import RmbData, find_rmb_files


BATCH_SIZE = 256
DEFAULT_SOURCE_KEY = "sd3_vae_left_128x96"
DEFAULT_OUTPUT_KEY = "sd3_vae_left_ae_12"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Add compact AE features derived from saved SD3 VAE latents."
    )
    parser.add_argument("path", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source_key", default=DEFAULT_SOURCE_KEY)
    parser.add_argument("--output_key", default=DEFAULT_OUTPUT_KEY)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def encode_episode(model, rmb_path, source_key, input_dim):
    with RmbData(rmb_path) as rmb_data:
        assert source_key in rmb_data, (rmb_path, source_key)
        source_features = rmb_data[source_key][:]
    assert source_features.ndim == 2, source_features.shape
    assert source_features.shape[1] == input_dim, source_features.shape

    features = []
    for start in range(0, len(source_features), BATCH_SIZE):
        batch = torch.from_numpy(
            source_features[start : start + BATCH_SIZE].astype(np.float32)
        ).cuda()
        with torch.inference_mode():
            features.append(model.encoder(batch).embedding.cpu())
    return torch.cat(features).numpy()


def main():
    args = parse_args()
    assert args.checkpoint.is_dir(), args.checkpoint
    assert torch.cuda.is_available(), "SD3 latent AE encoding requires a CUDA GPU."

    model = AutoModel.load_from_folder(str(args.checkpoint)).eval().cuda()
    model.requires_grad_(False)
    assert len(model.model_config.input_dim) == 1, model.model_config.input_dim
    input_dim = model.model_config.input_dim[0]
    latent_dim = model.model_config.latent_dim

    rmb_paths = find_rmb_files(str(args.path))
    assert rmb_paths, args.path
    for rmb_path in tqdm(rmb_paths, unit="episode"):
        features = encode_episode(model, rmb_path, args.source_key, input_dim)
        assert features.shape[1] == latent_dim, features.shape
        with RmbData(rmb_path, mode="r+") as rmb_data:
            if args.output_key in rmb_data:
                if not args.overwrite:
                    raise FileExistsError(f"{rmb_path}: {args.output_key}")
                del rmb_data.h5file[args.output_key]
            dataset = rmb_data.h5file.create_dataset(args.output_key, data=features)
            dataset.attrs["model"] = str(args.checkpoint)
            dataset.attrs["source_key"] = args.source_key
            dataset.attrs["input_dim"] = input_dim
            dataset.attrs["latent_dim"] = latent_dim

    print(f"HDF5 key: {args.output_key}")
    print(f"Feature shape per episode: (T, {latent_dim})")
    print(f"Episodes: {len(rmb_paths)}")


if __name__ == "__main__":
    main()
