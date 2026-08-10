import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from diffusers.models import AutoencoderKL


MODEL_NAME = "stabilityai/stable-diffusion-3-medium-diffusers"
DEFAULT_EPISODE = Path(
    "robo_manip_baselines/dataset/DatasetMujocoXarm7Pusht/"
    "WrenchPredObject0/WrenchPredObject0_world0_000.rmb"
)
IMAGE_WIDTH = 160
IMAGE_HEIGHT = 120


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reconstruct one DatasetMujocoXarm7Pusht left-camera episode with the SD3 VAE."
    )
    parser.add_argument("--episode", type=Path, default=DEFAULT_EPISODE)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs_to_user/SD3VAE_DatasetMujocoXarm7Pusht_left"),
    )
    parser.add_argument("--batch_size", type=int, default=16)
    return parser.parse_args()


def create_video_writer(path, fps, image_size):
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        image_size,
    )
    assert writer.isOpened(), f"Failed to open video writer: {path}"
    return writer


def add_comparison_labels(image):
    cv2.putText(
        image,
        "resized input",
        (6, 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        "resized input",
        (6, 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        "SD3 VAE reconstruction",
        (IMAGE_WIDTH + 6, 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        "SD3 VAE reconstruction",
        (IMAGE_WIDTH + 6, 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def encode_decode(vae, resized_rgb_images, device):
    image_tensor = torch.from_numpy(np.stack(resized_rgb_images)).to(
        device=device,
        dtype=torch.bfloat16,
    )
    image_tensor = image_tensor.permute(0, 3, 1, 2) / 127.5 - 1.0

    with torch.inference_mode():
        posterior = vae.encode(image_tensor).latent_dist
        latents = posterior.mode()
        latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor

        decode_latents = latents / vae.config.scaling_factor + vae.config.shift_factor
        reconstructed = vae.decode(decode_latents, return_dict=False)[0]

    reconstructed = ((reconstructed.float() + 1.0) / 2.0).clamp(0.0, 1.0)
    reconstructed = reconstructed.permute(0, 2, 3, 1).cpu().numpy()
    return latents.float().cpu().numpy(), reconstructed


def main():
    args = parse_args()
    assert args.batch_size > 0
    assert torch.cuda.is_available(), "This experiment requires a CUDA GPU."

    input_video_path = args.episode / "left_rgb_image.rmb.mp4"
    assert input_video_path.is_file(), f"Input video not found: {input_video_path}"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(input_video_path))
    assert capture.isOpened(), f"Failed to open input video: {input_video_path}"
    fps = capture.get(cv2.CAP_PROP_FPS)
    assert fps > 0
    expected_num_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    resized_video_path = args.output_dir / "left_resized_160x120.mp4"
    reconstructed_video_path = (
        args.output_dir / "left_SD3_VAE_reconstructed_160x120.mp4"
    )
    comparison_video_path = args.output_dir / "left_resized_vs_SD3_VAE.mp4"
    resized_writer = create_video_writer(
        resized_video_path, fps, (IMAGE_WIDTH, IMAGE_HEIGHT)
    )
    reconstructed_writer = create_video_writer(
        reconstructed_video_path, fps, (IMAGE_WIDTH, IMAGE_HEIGHT)
    )
    comparison_writer = create_video_writer(
        comparison_video_path, fps, (2 * IMAGE_WIDTH, IMAGE_HEIGHT)
    )

    device = torch.device("cuda")
    vae = AutoencoderKL.from_pretrained(
        MODEL_NAME,
        subfolder="vae",
        torch_dtype=torch.bfloat16,
    ).to(device)
    vae.eval().requires_grad_(False)

    latent_batches = []
    squared_error_sum = 0.0
    absolute_error_sum = 0.0
    num_values = 0
    num_frames = 0

    try:
        while True:
            resized_rgb_images = []
            for _ in range(args.batch_size):
                success, bgr_image = capture.read()
                if not success:
                    break
                resized_bgr_image = cv2.resize(
                    bgr_image,
                    (IMAGE_WIDTH, IMAGE_HEIGHT),
                    interpolation=cv2.INTER_AREA,
                )
                resized_rgb_images.append(
                    cv2.cvtColor(resized_bgr_image, cv2.COLOR_BGR2RGB)
                )

            if len(resized_rgb_images) == 0:
                break

            latents, reconstructed_rgb_images = encode_decode(
                vae, resized_rgb_images, device
            )
            assert latents.shape[1:] == (16, 15, 20), latents.shape
            latent_batches.append(latents.astype(np.float16))

            input_rgb_images = np.stack(resized_rgb_images).astype(np.float32) / 255.0
            difference = reconstructed_rgb_images - input_rgb_images
            squared_error_sum += float(np.square(difference).sum())
            absolute_error_sum += float(np.abs(difference).sum())
            num_values += difference.size

            for input_rgb, reconstructed_rgb in zip(
                resized_rgb_images, reconstructed_rgb_images
            ):
                input_bgr = cv2.cvtColor(input_rgb, cv2.COLOR_RGB2BGR)
                reconstructed_bgr = cv2.cvtColor(
                    np.round(255.0 * reconstructed_rgb).astype(np.uint8),
                    cv2.COLOR_RGB2BGR,
                )
                comparison_bgr = np.hstack((input_bgr, reconstructed_bgr))
                add_comparison_labels(comparison_bgr)

                resized_writer.write(input_bgr)
                reconstructed_writer.write(reconstructed_bgr)
                comparison_writer.write(comparison_bgr)

            num_frames += len(resized_rgb_images)
            print(f"Processed {num_frames}/{expected_num_frames} frames")
    finally:
        capture.release()
        resized_writer.release()
        reconstructed_writer.release()
        comparison_writer.release()

    assert num_frames == expected_num_frames, (num_frames, expected_num_frames)
    latents = np.concatenate(latent_batches, axis=0)
    assert latents.shape == (num_frames, 16, 15, 20), latents.shape
    latents = np.transpose(latents, (0, 2, 3, 1))
    latent_path = args.output_dir / "left_SD3_VAE_latents_mode_float16.npy"
    np.save(latent_path, latents)

    mse = squared_error_sum / num_values
    summary = {
        "model": MODEL_NAME,
        "source_episode": str(args.episode),
        "source_video": str(input_video_path),
        "camera": "left",
        "num_frames": num_frames,
        "fps": fps,
        "resized_rgb_shape": [num_frames, IMAGE_HEIGHT, IMAGE_WIDTH, 3],
        "latent_shape": list(latents.shape),
        "latent_dtype": str(latents.dtype),
        "latent_distribution_value": "mode",
        "mse": mse,
        "mae": absolute_error_sum / num_values,
        "psnr_db": float("inf") if mse == 0.0 else float(-10.0 * np.log10(mse)),
        "resized_video": str(resized_video_path),
        "reconstructed_video": str(reconstructed_video_path),
        "comparison_video": str(comparison_video_path),
        "latent_file": str(latent_path),
    }
    summary_path = args.output_dir / "summary.json"
    with summary_path.open("w") as file:
        json.dump(summary, file, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
