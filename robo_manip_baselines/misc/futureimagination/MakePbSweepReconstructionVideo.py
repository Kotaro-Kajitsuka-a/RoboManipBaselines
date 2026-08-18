import argparse
import pickle
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch
from pythae.models import AutoModel
from torch.utils.data import DataLoader

from robo_manip_baselines.common import DataKey, RmbData, denormalize_data
from robo_manip_baselines.policy.wrench_predictor4.EvalWrenchPredictor4SweepCommon import (
    EvalWrenchPredictor4Dataset,
)
from robo_manip_baselines.policy.wrench_predictor4.WrenchPredictor4Model import (
    WrenchPredictor4Model,
)


BATCH_SIZE = 64
DEFAULT_MATERIAL_OBJECT_IDS = (0, 1, 2)


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Create one video comparing the real camera image with ImageVAE "
            "reconstructions predicted using each WrenchPredictor4 PB."
        ),
    )
    parser.add_argument(
        "checkpoint_dir",
        type=Path,
        help="WrenchPredictor4 checkpoint directory",
    )
    parser.add_argument("rmb_path", type=Path, help="one RMB episode directory")
    parser.add_argument(
        "vae_checkpoint",
        type=Path,
        help="ImageVAE checkpoint directory containing model.pt",
    )
    parser.add_argument(
        "--checkpoint_name",
        default="policy_best.ckpt",
        help="WrenchPredictor4 checkpoint filename",
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
    return parser.parse_args()


def infer_camera_name(image_feature_key, latent_dim):
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
    policy = WrenchPredictor4Model(**model_meta_info["policy"]["args"])
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
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
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


def decode_features(vae, features, device):
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
    assert args.vae_checkpoint.is_dir(), args.vae_checkpoint
    assert len(args.material_object_ids) == len(set(args.material_object_ids))
    assert torch.cuda.is_available(), "PB reconstruction requires a CUDA GPU."
    device = torch.device("cuda")

    model_meta_info = load_model_meta_info(args.checkpoint_dir)
    image_feature_dim = model_meta_info["policy"]["args"]["image_feature_dim"]
    image_feature_key = model_meta_info["data"]["image_feature_key"]
    vae = AutoModel.load_from_folder(str(args.vae_checkpoint)).eval().to(device)
    vae.requires_grad_(False)
    assert vae.model_config.latent_dim == image_feature_dim, (
        vae.model_config.latent_dim,
        image_feature_dim,
    )

    camera_name = args.camera_name
    if camera_name is None:
        camera_name = infer_camera_name(image_feature_key, image_feature_dim)
    video_path = args.rmb_path / f"{camera_name}_rgb_image.rmb.mp4"
    assert video_path.is_file(), video_path

    object_key_to_id = model_meta_info["material_property"]["object_key_to_id"]
    known_object_ids = set(object_key_to_id.values())
    assert all(
        material_object_id in known_object_ids
        for material_object_id in args.material_object_ids
    ), args.material_object_ids
    checkpoint_meta = model_meta_info["material_property"][
        Path(args.checkpoint_name).stem
    ]
    material_pb_by_id = {
        object_id: checkpoint_meta["pb_by_object"][f"WrenchPredObject{object_id}"][0]
        for object_id in args.material_object_ids
    }

    policy = load_policy(
        args.checkpoint_dir,
        args.checkpoint_name,
        model_meta_info,
        device,
    )
    raw_frame_idx, material_id_to_features = predict_image_features(
        policy,
        model_meta_info,
        args.rmb_path,
        args.material_object_ids,
        device,
    )
    reconstructed_by_material_id = {
        material_object_id: decode_features(vae, features, device)
        for material_object_id, features in material_id_to_features.items()
    }
    input_frames, source_fps, width, height = read_selected_frames(
        video_path,
        raw_frame_idx,
    )

    output_path = args.output
    if output_path is None:
        output_path = args.rmb_path.parent / (
            f"{args.rmb_path.stem}_{camera_name}_pb_reconstruction.mp4"
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

    print(f"RMB: {args.rmb_path}")
    print(f"Checkpoint: {args.checkpoint_dir / args.checkpoint_name}")
    print(f"Image feature: {image_feature_key} ({image_feature_dim}D)")
    print(f"Camera: {camera_name}")
    print(f"Frames: {len(input_frames)} at {source_fps / skip:.3f} fps")
    print(f"Output: {output_path.resolve()}")


if __name__ == "__main__":
    main()
