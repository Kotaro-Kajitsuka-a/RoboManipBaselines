import argparse
import pickle
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch
from diffusers.models import AutoencoderKL
from pythae.models import AutoModel
from torch.utils.data import DataLoader

from robo_manip_baselines.common import DataKey, RmbData, denormalize_data
from robo_manip_baselines.common import find_rmb_files
from robo_manip_baselines.policy.wrench_predictor4.EvalWrenchPredictor4SweepCommon import (
    EvalWrenchPredictor4Dataset,
)
from robo_manip_baselines.policy.wrench_predictor4.WrenchPredictor4Model import (
    WrenchPredictor4Model,
)
from robo_manip_baselines.policy.wrench_predictor5.WrenchPredictor5Model import (
    WrenchPredictor5Model,
)


BATCH_SIZE = 64
WP5_BATCH_SIZE = 8
DEFAULT_MATERIAL_OBJECT_IDS = (0, 1, 2)
SD3_MODEL_NAME = "stabilityai/stable-diffusion-3-medium-diffusers"
SD3_LATENT_SHAPE = (16, 12, 16)
SD3_LATENT_DIM = int(np.prod(SD3_LATENT_SHAPE))


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Create one video comparing the real camera image with VAE "
            "reconstructions predicted using each WrenchPredictor PB."
        ),
    )
    parser.add_argument(
        "checkpoint_dir",
        type=Path,
        help="WrenchPredictor4 or WrenchPredictor5 checkpoint directory",
    )
    parser.add_argument(
        "rmb_path",
        type=Path,
        help="one RMB episode directory or a dataset directory",
    )
    parser.add_argument(
        "vae_checkpoint",
        type=Path,
        nargs="?",
        default=None,
        help="ImageVAE checkpoint directory (required only for WrenchPredictor4)",
    )
    parser.add_argument(
        "--checkpoint_name",
        default="policy_best.ckpt",
        help="WrenchPredictor checkpoint filename",
    )
    parser.add_argument(
        "--material_object_ids",
        type=int,
        nargs="+",
        default=DEFAULT_MATERIAL_OBJECT_IDS,
        help="PB object IDs shown in the video",
    )
    parser.add_argument("--camera_name", default=None, help="override camera name")
    parser.add_argument("--output", type=Path, default=None, help="output MP4 path")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="output directory when rmb_path is a dataset directory",
    )
    return parser.parse_args()


def infer_camera_name(image_feature_key, latent_dim, policy_name):
    if policy_name == "WrenchPredictor5":
        if image_feature_key == "sd3_vae":
            return "left"
        prefix = "sd3_vae_"
        assert image_feature_key.startswith(prefix), image_feature_key
        camera_name = image_feature_key[len(prefix) :]
        assert camera_name, image_feature_key
        return camera_name

    assert policy_name == "WrenchPredictor4", policy_name
    sd3_prefix = "sd3_vae_"
    sd3_suffix = f"_ae_{latent_dim}"
    if image_feature_key.startswith(sd3_prefix) and image_feature_key.endswith(
        sd3_suffix
    ):
        camera_name = image_feature_key[len(sd3_prefix) : -len(sd3_suffix)]
        assert camera_name, image_feature_key
        return camera_name

    prefix = "image_vae_"
    suffix = f"_{latent_dim}"
    assert image_feature_key.startswith(prefix), image_feature_key
    assert image_feature_key.endswith(suffix), image_feature_key
    camera_name = image_feature_key[len(prefix) : -len(suffix)]
    assert camera_name, image_feature_key
    return camera_name


def add_label(image, text, x, y):
    cv2.putText(
        image,
        text,
        (x + 12, y + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text,
        (x + 12, y + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def load_model_meta_info(checkpoint_dir):
    path = checkpoint_dir / "model_meta_info.pkl"
    assert path.is_file(), path
    with path.open("rb") as file:
        return pickle.load(file)


def load_policy(checkpoint_dir, checkpoint_name, model_meta_info, device):
    checkpoint = checkpoint_dir / checkpoint_name
    assert checkpoint.is_file(), checkpoint
    policy_name = model_meta_info["policy"]["name"]
    if policy_name == "WrenchPredictor4":
        policy_class = WrenchPredictor4Model
    else:
        assert policy_name == "WrenchPredictor5", policy_name
        policy_class = WrenchPredictor5Model
    policy = policy_class(**model_meta_info["policy"]["args"])
    policy.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True)
    )
    policy.to(device).eval().requires_grad_(False)
    return policy


def predict_image_features(
    policy,
    model_meta_info,
    rmb_path,
    material_object_ids,
    device,
):
    dataset = EvalWrenchPredictor4Dataset(
        [str(rmb_path)],
        model_meta_info,
        enable_rmb_cache=False,
    )
    batch_size = (
        WP5_BATCH_SIZE
        if model_meta_info["policy"]["name"] == "WrenchPredictor5"
        else BATCH_SIZE
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    horizon = model_meta_info["data"]["horizon"]
    skip = model_meta_info["data"]["skip"]
    with RmbData(str(rmb_path)) as rmb_data:
        episode_len = rmb_data[DataKey.TIME][::skip].shape[0]
    time_idx = np.asarray(
        [
            np.clip(start_time_idx + horizon - 1, 0, episode_len - 1)
            for _episode_idx, start_time_idx in dataset.chunk_info_list
        ],
        dtype=np.int64,
    )
    assert len(time_idx) == len(np.unique(time_idx)), time_idx

    material_id_to_features = {}
    for material_object_id in material_object_ids:
        predicted_batches = []
        for batch in dataloader:
            batch = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            batch["object_id"] = torch.full_like(
                batch["object_id"], material_object_id
            )
            with torch.inference_mode():
                prediction = policy.predict(batch)["image_feature"][:, -1]
            predicted_batches.append(prediction.cpu().numpy())
        normalized_features = np.concatenate(predicted_batches)
        material_id_to_features[material_object_id] = denormalize_data(
            normalized_features,
            model_meta_info["image_feature"],
        )

    return time_idx * skip, material_id_to_features


def decode_image_vae_features(vae, features, device):
    reconstructed_batches = []
    for start in range(0, len(features), BATCH_SIZE):
        latent = torch.from_numpy(features[start : start + BATCH_SIZE]).to(
            device=device,
            dtype=torch.float32,
        )
        with torch.inference_mode():
            reconstruction = vae.decoder(latent).reconstruction
        reconstructed_batches.append(
            reconstruction.permute(0, 2, 3, 1).cpu().numpy()
        )
    reconstructed = np.concatenate(reconstructed_batches)
    return np.round(255.0 * reconstructed).clip(0, 255).astype(np.uint8)


def decode_sd3_features(vae, features, latent_shape, device):
    reconstructed_batches = []
    for start in range(0, len(features), WP5_BATCH_SIZE):
        latent = torch.from_numpy(features[start : start + WP5_BATCH_SIZE]).to(
            device=device,
            dtype=torch.bfloat16,
        )
        latent = latent.reshape(-1, *latent_shape)
        decode_latent = latent / vae.config.scaling_factor + vae.config.shift_factor
        with torch.inference_mode():
            reconstruction = vae.decode(decode_latent, return_dict=False)[0]
        reconstruction = ((reconstruction.float() + 1.0) / 2.0).clamp(0.0, 1.0)
        reconstructed_batches.append(
            reconstruction.permute(0, 2, 3, 1).cpu().numpy()
        )
    reconstructed = np.concatenate(reconstructed_batches)
    return np.round(255.0 * reconstructed).astype(np.uint8)


def decode_sd3_latent_ae_features(sd3_vae, latent_ae, features, device):
    reconstructed_batches = []
    for start in range(0, len(features), WP5_BATCH_SIZE):
        compact_latent = torch.from_numpy(
            features[start : start + WP5_BATCH_SIZE]
        ).to(device=device, dtype=torch.float32)
        with torch.inference_mode():
            flat_sd3_latent = latent_ae.decoder(compact_latent).reconstruction
        assert flat_sd3_latent.shape[1] == SD3_LATENT_DIM, flat_sd3_latent.shape
        sd3_latent = flat_sd3_latent.reshape(-1, *SD3_LATENT_SHAPE)
        reconstructed_batches.append(
            decode_sd3_features(
                sd3_vae,
                sd3_latent.cpu().numpy(),
                SD3_LATENT_SHAPE,
                device,
            )
        )
    return np.concatenate(reconstructed_batches)


def read_selected_frames(video_path, raw_frame_idx):
    capture = cv2.VideoCapture(str(video_path))
    assert capture.isOpened(), video_path
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    selected_frames = []
    next_selected_offset = 0
    frame_idx = 0
    while next_selected_offset < len(raw_frame_idx):
        success, frame = capture.read()
        if not success:
            break
        if frame_idx == raw_frame_idx[next_selected_offset]:
            selected_frames.append(frame)
            next_selected_offset += 1
        frame_idx += 1
    capture.release()
    assert len(selected_frames) == len(raw_frame_idx), (
        video_path,
        len(selected_frames),
        len(raw_frame_idx),
    )
    return selected_frames, fps, width, height


def encode_video(
    output_path,
    input_frames,
    reconstructed_by_material_id,
    material_object_ids,
    material_pb_by_id,
    fps,
    width,
    height,
    camera_name,
):
    assert len(material_object_ids) == 3, material_object_ids
    output_path.parent.mkdir(parents=True, exist_ok=True)
    work_path = output_path.with_suffix(".mp4v.mp4")
    encoded_path = output_path.with_suffix(".h264.mp4")
    writer = cv2.VideoWriter(
        str(work_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (2 * width, 2 * height),
    )
    assert writer.isOpened(), work_path

    try:
        for frame_offset, input_frame in enumerate(input_frames):
            panels = [input_frame]
            for material_object_id in material_object_ids:
                reconstructed_rgb = reconstructed_by_material_id[material_object_id][
                    frame_offset
                ]
                reconstructed_bgr = cv2.cvtColor(
                    reconstructed_rgb,
                    cv2.COLOR_RGB2BGR,
                )
                panels.append(
                    cv2.resize(
                        reconstructed_bgr,
                        (width, height),
                        interpolation=cv2.INTER_LINEAR,
                    )
                )
            canvas = np.vstack((np.hstack(panels[:2]), np.hstack(panels[2:])))
            add_label(canvas, f"actual {camera_name}", 0, 0)
            panel_origins = ((width, 0), (0, height), (width, height))
            for material_object_id, (x, y) in zip(
                material_object_ids,
                panel_origins,
                strict=True,
            ):
                pb = material_pb_by_id[material_object_id]
                add_label(
                    canvas,
                    f"pred Object{material_object_id} PB={pb:.4f}",
                    x,
                    y,
                )
            writer.write(canvas)
    finally:
        writer.release()

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
    assert args.checkpoint_dir.is_dir(), args.checkpoint_dir
    assert args.rmb_path.is_dir(), args.rmb_path
    assert len(args.material_object_ids) == len(set(args.material_object_ids))
    if args.rmb_path.suffix == ".rmb":
        assert args.output_dir is None
        rmb_paths = [args.rmb_path]
    else:
        assert args.output is None, "Use --output_dir for a dataset directory."
        rmb_paths = [Path(path) for path in find_rmb_files(str(args.rmb_path))]
        assert rmb_paths, args.rmb_path
    assert torch.cuda.is_available(), "PB reconstruction requires a CUDA GPU."
    device = torch.device("cuda")

    model_meta_info = load_model_meta_info(args.checkpoint_dir)
    policy_name = model_meta_info["policy"]["name"]
    image_feature_dim = model_meta_info["policy"]["args"]["image_feature_dim"]
    image_feature_key = model_meta_info["data"]["image_feature_key"]
    sd3_vae = None
    latent_ae = None
    if policy_name == "WrenchPredictor4":
        assert args.vae_checkpoint is not None
        assert args.vae_checkpoint.is_dir(), args.vae_checkpoint
        vae = AutoModel.load_from_folder(str(args.vae_checkpoint)).eval().to(device)
        vae.requires_grad_(False)
        assert vae.model_config.latent_dim == image_feature_dim, (
            vae.model_config.latent_dim,
            image_feature_dim,
        )
        if tuple(vae.model_config.input_dim) == (SD3_LATENT_DIM,):
            latent_ae = vae
            sd3_vae = AutoencoderKL.from_pretrained(
                SD3_MODEL_NAME,
                subfolder="vae",
                torch_dtype=torch.bfloat16,
                use_safetensors=True,
            ).to(device)
            sd3_vae.eval().requires_grad_(False)
    else:
        assert policy_name == "WrenchPredictor5", policy_name
        assert args.vae_checkpoint is None, (
            "WrenchPredictor5 uses the pretrained SD3 VAE; do not pass "
            "vae_checkpoint"
        )
        vae = AutoencoderKL.from_pretrained(
            SD3_MODEL_NAME,
            subfolder="vae",
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
        ).to(device)
        vae.eval().requires_grad_(False)

    camera_name = args.camera_name
    if camera_name is None:
        camera_name = infer_camera_name(
            image_feature_key,
            image_feature_dim,
            policy_name,
        )
    object_key_to_id = model_meta_info["material_property"]["object_key_to_id"]
    known_object_ids = set(object_key_to_id.values())
    assert all(
        material_object_id in known_object_ids
        for material_object_id in args.material_object_ids
    ), args.material_object_ids
    checkpoint = args.checkpoint_dir / args.checkpoint_name
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    material_property = state_dict["material_property.weight"]
    assert material_property.shape[1] == 1, material_property.shape
    material_pb_by_id = {
        object_id: material_property[object_id, 0].item()
        for object_id in args.material_object_ids
    }

    policy = load_policy(
        args.checkpoint_dir,
        args.checkpoint_name,
        model_meta_info,
        device,
    )
    print(f"Checkpoint: {args.checkpoint_dir / args.checkpoint_name}")
    print(f"Policy: {policy_name}")
    print(f"Image feature: {image_feature_key} ({image_feature_dim}D)")
    print(f"Camera: {camera_name}")
    print(f"Episodes: {len(rmb_paths)}")

    output_dir = args.output_dir
    if len(rmb_paths) > 1 and output_dir is None:
        output_dir = args.rmb_path.with_name(
            f"{args.rmb_path.name}_PbReconstructionVideos"
        )

    for episode_idx, rmb_path in enumerate(rmb_paths, start=1):
        video_path = rmb_path / f"{camera_name}_rgb_image.rmb.mp4"
        assert video_path.is_file(), video_path
        raw_frame_idx, material_id_to_features = predict_image_features(
            policy,
            model_meta_info,
            rmb_path,
            args.material_object_ids,
            device,
        )
        if latent_ae is not None:
            reconstructed_by_material_id = {
                material_object_id: decode_sd3_latent_ae_features(
                    sd3_vae,
                    latent_ae,
                    features,
                    device,
                )
                for material_object_id, features in material_id_to_features.items()
            }
        elif policy_name == "WrenchPredictor4":
            reconstructed_by_material_id = {
                material_object_id: decode_image_vae_features(vae, features, device)
                for material_object_id, features in material_id_to_features.items()
            }
        else:
            latent_shape = tuple(model_meta_info["policy"]["args"]["latent_shape"])
            reconstructed_by_material_id = {
                material_object_id: decode_sd3_features(
                    vae,
                    features,
                    latent_shape,
                    device,
                )
                for material_object_id, features in material_id_to_features.items()
            }
        input_frames, source_fps, width, height = read_selected_frames(
            video_path,
            raw_frame_idx,
        )

        if output_dir is not None:
            output_path = output_dir / (
                f"{rmb_path.stem}_{camera_name}_pb_reconstruction.mp4"
            )
        elif args.output is not None:
            output_path = args.output
        else:
            output_path = rmb_path.parent / (
                f"{rmb_path.stem}_{camera_name}_pb_reconstruction.mp4"
            )
        skip = model_meta_info["data"]["skip"]
        encode_video(
            output_path,
            input_frames,
            reconstructed_by_material_id,
            args.material_object_ids,
            material_pb_by_id,
            source_fps / skip,
            width,
            height,
            camera_name,
        )
        print(
            f"[{episode_idx}/{len(rmb_paths)}] {rmb_path.name}: "
            f"{len(input_frames)} frames -> {output_path.resolve()}"
        )


if __name__ == "__main__":
    main()
