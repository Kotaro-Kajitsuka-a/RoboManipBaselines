import argparse
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from robo_manip_baselines.common import (
    RmbData,
    convert_data_to_policy,
    find_rmb_files,
)
from robo_manip_baselines.misc.futureimagination.ImageJointConditionedCVAE import (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    load_image_joint_conditioned_cvae,
)
from robo_manip_baselines.misc.futureimagination.MakeReconstructionVideos import (
    add_label,
)

BATCH_SIZE = 64
OUTPUT_SUFFIX = "ReconCvaeVideos"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create side-by-side original and joint-conditioned CVAE "
            "reconstruction videos."
        )
    )
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--camera_name", required=True)
    return parser.parse_args()


def load_condition(rmb_path, condition_keys):
    with RmbData(rmb_path) as rmb_data:
        return np.concatenate(
            [convert_data_to_policy(rmb_data[key][:], key) for key in condition_keys],
            axis=1,
        ).astype(np.float32)


def reconstruct(model, input_bgr_images, condition):
    resized_rgb_images = np.stack(
        [
            cv2.cvtColor(
                cv2.resize(
                    image,
                    (IMAGE_WIDTH, IMAGE_HEIGHT),
                    interpolation=cv2.INTER_LINEAR,
                ),
                cv2.COLOR_BGR2RGB,
            )
            for image in input_bgr_images
        ]
    )
    images = (
        torch.from_numpy(resized_rgb_images).cuda().permute(0, 3, 1, 2).float() / 255.0
    )
    condition = torch.from_numpy(condition).cuda()
    with torch.inference_mode():
        reconstructed = model.reconstruct(images, condition)
    reconstructed = reconstructed.permute(0, 2, 3, 1).cpu().numpy()
    return np.round(255.0 * reconstructed).clip(0, 255).astype(np.uint8)


def create_episode_video(model, rmb_path, output_dir, camera_name):
    input_path = Path(rmb_path) / f"{camera_name}_rgb_image.rmb.mp4"
    assert input_path.is_file(), input_path
    output_path = output_dir / (
        f"{Path(rmb_path).stem}_{camera_name}_vs_cvae{model.latent_dim}.mp4"
    )
    work_path = output_path.with_suffix(".mp4v.mp4")
    encoded_path = output_path.with_suffix(".h264.mp4")
    condition = load_condition(rmb_path, model.condition_keys)

    capture = cv2.VideoCapture(str(input_path))
    assert capture.isOpened(), input_path
    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    assert fps > 0.0 and frame_count > 0, input_path
    assert frame_count == len(condition), (input_path, frame_count, len(condition))

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

            end = written_frame_count + len(input_images)
            reconstructed_rgb_images = reconstruct(
                model,
                input_images,
                condition[written_frame_count:end],
            )
            for input_image, reconstructed_rgb in zip(
                input_images,
                reconstructed_rgb_images,
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
                add_label(
                    comparison,
                    f"CVAE{model.latent_dim} reconstruction",
                    width,
                )
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
    assert args.checkpoint.is_dir(), args.checkpoint
    assert torch.cuda.is_available(), "Image CVAE reconstruction requires a CUDA GPU."
    model = load_image_joint_conditioned_cvae(args.checkpoint, device="cuda").eval()
    model.requires_grad_(False)
    rmb_paths = find_rmb_files(str(args.dataset_dir))
    assert rmb_paths, args.dataset_dir
    output_dir = args.dataset_dir.with_name(f"{args.dataset_dir.name}_{OUTPUT_SUFFIX}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset: {args.dataset_dir}")
    print(f"CVAE checkpoint: {args.checkpoint}")
    print(f"Camera: {args.camera_name}")
    print(f"Condition keys: {model.condition_keys}")
    print(f"Output: {output_dir}")
    for rmb_path in tqdm(rmb_paths, unit="episode"):
        create_episode_video(model, rmb_path, output_dir, args.camera_name)


if __name__ == "__main__":
    main()
