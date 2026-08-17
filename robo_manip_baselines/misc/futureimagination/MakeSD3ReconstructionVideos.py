import argparse
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch
from diffusers.models import AutoencoderKL
from tqdm import tqdm

from robo_manip_baselines.common import find_rmb_files


MODEL_NAME = "stabilityai/stable-diffusion-3-medium-diffusers"
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 96
BATCH_SIZE = 16
OUTPUT_SUFFIX = "ReconVaeVideosSD3"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create original and SD3 VAE reconstruction comparison videos."
    )
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--camera_name", required=True)
    return parser.parse_args()


def add_label(image, text, x):
    cv2.putText(
        image,
        text,
        (x + 6, 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text,
        (x + 6, 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def reconstruct(vae, resized_rgb_images, latent_shape):
    images = torch.from_numpy(np.stack(resized_rgb_images)).to(
        device="cuda",
        dtype=torch.bfloat16,
    )
    images = images.permute(0, 3, 1, 2) / 127.5 - 1.0
    with torch.inference_mode():
        latents = vae.encode(images).latent_dist.mode()
        latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor
        decode_latents = latents / vae.config.scaling_factor + vae.config.shift_factor
        reconstructed = vae.decode(decode_latents, return_dict=False)[0]
    assert latents.shape[1:] == latent_shape, latents.shape
    reconstructed = ((reconstructed.float() + 1.0) / 2.0).clamp(0.0, 1.0)
    reconstructed = reconstructed.permute(0, 2, 3, 1).cpu().numpy()
    return np.round(255.0 * reconstructed).astype(np.uint8)


def create_episode_video(
    vae,
    rmb_path,
    output_dir,
    camera_name,
    image_size,
    latent_shape,
):
    image_width, image_height = image_size
    display_size = image_size
    input_path = Path(rmb_path) / f"{camera_name}_rgb_image.rmb.mp4"
    assert input_path.is_file(), input_path
    output_path = output_dir / f"{Path(rmb_path).stem}_{camera_name}_vs_sd3_vae.mp4"
    work_path = output_path.with_suffix(".mp4v.mp4")
    encoded_path = output_path.with_suffix(".h264.mp4")

    capture = cv2.VideoCapture(str(input_path))
    assert capture.isOpened(), input_path
    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    assert fps > 0.0 and frame_count > 0, input_path

    writer = cv2.VideoWriter(
        str(work_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (2 * display_size[0], display_size[1]),
    )
    assert writer.isOpened(), work_path

    written_frame_count = 0
    try:
        while True:
            resized_rgb_images = []
            for _ in range(BATCH_SIZE):
                success, bgr_image = capture.read()
                if not success:
                    break
                resized_bgr = cv2.resize(
                    bgr_image,
                    image_size,
                    interpolation=cv2.INTER_LINEAR,
                )
                resized_rgb_images.append(cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB))
            if not resized_rgb_images:
                break

            reconstructed_rgb_images = reconstruct(
                vae,
                resized_rgb_images,
                latent_shape,
            )
            for input_rgb, reconstructed_rgb in zip(
                resized_rgb_images, reconstructed_rgb_images
            ):
                input_bgr = cv2.cvtColor(input_rgb, cv2.COLOR_RGB2BGR)
                reconstructed_bgr = cv2.cvtColor(
                    reconstructed_rgb,
                    cv2.COLOR_RGB2BGR,
                )
                input_bgr = cv2.resize(
                    input_bgr,
                    display_size,
                    interpolation=cv2.INTER_NEAREST,
                )
                reconstructed_bgr = cv2.resize(
                    reconstructed_bgr,
                    display_size,
                    interpolation=cv2.INTER_NEAREST,
                )
                comparison = np.hstack((input_bgr, reconstructed_bgr))
                add_label(comparison, f"resized {camera_name}", 0)
                add_label(comparison, "SD3 VAE reconstruction", display_size[0])
                writer.write(comparison)
                written_frame_count += 1
    finally:
        capture.release()
        writer.release()

    assert written_frame_count == frame_count, (
        input_path,
        written_frame_count,
        frame_count,
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(work_path),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(encoded_path),
        ],
        check=True,
    )
    encoded_path.replace(output_path)
    work_path.unlink()


def main():
    args = parse_args()
    assert args.dataset_dir.is_dir(), args.dataset_dir
    assert torch.cuda.is_available(), "SD3 VAE reconstruction requires a CUDA GPU."

    rmb_paths = find_rmb_files(str(args.dataset_dir))
    assert rmb_paths, f"No RMB episodes found in {args.dataset_dir}"
    image_size = (IMAGE_WIDTH, IMAGE_HEIGHT)
    latent_shape = (16, IMAGE_HEIGHT // 8, IMAGE_WIDTH // 8)
    output_dir = args.dataset_dir.with_name(
        f"{args.dataset_dir.name}_{OUTPUT_SUFFIX}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    vae = AutoencoderKL.from_pretrained(
        MODEL_NAME,
        subfolder="vae",
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
    ).cuda()
    vae.eval().requires_grad_(False)

    print(f"Dataset: {args.dataset_dir}")
    print(f"SD3 VAE: {MODEL_NAME}")
    print(f"Camera: {args.camera_name}")
    print(f"Input size: {IMAGE_WIDTH}x{IMAGE_HEIGHT}")
    print(f"Latent shape: {latent_shape}")
    print(f"Output: {output_dir}")
    for rmb_path in tqdm(rmb_paths, unit="episode"):
        create_episode_video(
            vae,
            rmb_path,
            output_dir,
            args.camera_name,
            image_size,
            latent_shape,
        )


if __name__ == "__main__":
    main()
