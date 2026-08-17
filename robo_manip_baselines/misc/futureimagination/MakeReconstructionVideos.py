import argparse
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch
from pythae.models import AutoModel
from tqdm import tqdm

from robo_manip_baselines.common import find_rmb_files


BATCH_SIZE = 64
OUTPUT_SUFFIX = "ReconVaeVideos"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create side-by-side original and ImageVAE reconstruction videos."
    )
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("vae_checkpoint", type=Path)
    parser.add_argument("--camera_name", required=True)
    return parser.parse_args()


def add_label(image, text, x):
    cv2.putText(
        image,
        text,
        (x + 12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text,
        (x + 12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def reconstruct(model, input_bgr_images, image_size):
    resized_rgb_images = np.stack(
        [
            cv2.cvtColor(
                cv2.resize(image, image_size, interpolation=cv2.INTER_LINEAR),
                cv2.COLOR_BGR2RGB,
            )
            for image in input_bgr_images
        ]
    )
    images = (
        torch.from_numpy(resized_rgb_images).cuda().permute(0, 3, 1, 2).float() / 255.0
    )
    with torch.inference_mode():
        latents = model.encoder(images).embedding
        reconstructed = model.decoder(latents).reconstruction
    reconstructed = reconstructed.permute(0, 2, 3, 1).cpu().numpy()
    return np.round(255.0 * reconstructed).clip(0, 255).astype(np.uint8)


def create_episode_video(
    model,
    rmb_path,
    output_dir,
    camera_name,
    latent_dim,
    image_size,
):
    input_path = Path(rmb_path) / f"{camera_name}_rgb_image.rmb.mp4"
    assert input_path.is_file(), input_path
    output_path = (
        output_dir / f"{Path(rmb_path).stem}_{camera_name}_vs_vae{latent_dim}.mp4"
    )
    work_path = output_path.with_suffix(".mp4v.mp4")
    encoded_path = output_path.with_suffix(".h264.mp4")

    capture = cv2.VideoCapture(str(input_path))
    assert capture.isOpened(), input_path
    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    assert fps > 0.0 and frame_count > 0, input_path

    writer = cv2.VideoWriter(
        str(work_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (2 * width, height),
    )
    assert writer.isOpened(), work_path

    written_frame_count = 0
    try:
        while True:
            input_images = []
            for _ in range(BATCH_SIZE):
                success, image = capture.read()
                if not success:
                    break
                input_images.append(image)
            if not input_images:
                break

            reconstructed_rgb_images = reconstruct(model, input_images, image_size)
            for input_image, reconstructed_rgb in zip(
                input_images, reconstructed_rgb_images
            ):
                reconstructed_bgr = cv2.cvtColor(
                    reconstructed_rgb,
                    cv2.COLOR_RGB2BGR,
                )
                reconstructed_bgr = cv2.resize(
                    reconstructed_bgr,
                    (width, height),
                    interpolation=cv2.INTER_LINEAR,
                )
                comparison = np.hstack((input_image, reconstructed_bgr))
                add_label(comparison, f"original {camera_name} camera", 0)
                add_label(comparison, f"VAE{latent_dim} reconstruction", width)
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
    assert args.vae_checkpoint.is_dir(), args.vae_checkpoint
    assert torch.cuda.is_available(), "VAE reconstruction requires a CUDA GPU."

    rmb_paths = find_rmb_files(str(args.dataset_dir))
    assert rmb_paths, f"No RMB episodes found in {args.dataset_dir}"

    output_dir = args.dataset_dir.with_name(f"{args.dataset_dir.name}_{OUTPUT_SUFFIX}")
    output_dir.mkdir(parents=True, exist_ok=True)

    model = AutoModel.load_from_folder(str(args.vae_checkpoint)).eval().cuda()
    model.requires_grad_(False)
    _, image_height, image_width = model.model_config.input_dim
    latent_dim = model.model_config.latent_dim

    print(f"Dataset: {args.dataset_dir}")
    print(f"VAE checkpoint: {args.vae_checkpoint}")
    print(f"Camera: {args.camera_name}")
    print(f"Output: {output_dir}")
    for rmb_path in tqdm(rmb_paths, unit="episode"):
        create_episode_video(
            model,
            rmb_path,
            output_dir,
            args.camera_name,
            latent_dim,
            (image_width, image_height),
        )


if __name__ == "__main__":
    main()
