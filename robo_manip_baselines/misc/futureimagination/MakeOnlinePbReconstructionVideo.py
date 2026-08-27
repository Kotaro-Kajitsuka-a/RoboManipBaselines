import argparse
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch
from pythae.models import AutoModel

from robo_manip_baselines.common import (
    DataKey,
    RmbData,
    convert_data_to_policy,
    denormalize_data,
    find_rmb_files,
    get_skipped_data_seq,
    normalize_data,
)
from robo_manip_baselines.policy.wrench_predictor4_online.WrenchPredictor4OnlineUtils import (
    load_model_meta_info,
    load_pb_table,
    load_policy,
)


BATCH_SIZE = 64
REFERENCE_OBJECT_IDS = (0, 1, 2)
DEFAULT_OUTPUT_SIZE = (640, 480)
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Create five-panel videos comparing the actual camera image with the "
            "farthest WP4 prediction from the online PB and learned PB0/PB1/PB2. "
            "The WP4 checkpoint is read from each RMB episode."
        ),
    )
    parser.add_argument(
        "rmb_path",
        type=Path,
        help="one online-PB RMB episode or a directory containing episodes",
    )
    parser.add_argument(
        "--vae_checkpoint",
        type=Path,
        default=None,
        help=(
            "ImageVAE checkpoint override; normally read from RMB metadata or the "
            "stored image-feature attributes"
        ),
    )
    parser.add_argument(
        "--camera_name",
        default=None,
        help="camera override; normally read from RMB or image-feature metadata",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="output MP4 path; valid only for one RMB episode",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="output directory when rmb_path contains multiple episodes",
    )
    parser.add_argument(
        "--output_width",
        type=int,
        default=DEFAULT_OUTPUT_SIZE[0],
        help="assembled video width",
    )
    parser.add_argument(
        "--output_height",
        type=int,
        default=DEFAULT_OUTPUT_SIZE[1],
        help="assembled video height",
    )
    return parser.parse_args()


def resolve_recorded_path(value) -> Path:
    if isinstance(value, bytes):
        value = value.decode()

    recorded_path = Path(str(value)).expanduser()
    resolved_path = recorded_path.resolve()

    if resolved_path.exists():
        return resolved_path

    parts = recorded_path.parts
    marker = ("robo_manip_baselines", "checkpoint")

    for part_idx in range(len(parts) - 1):
        if parts[part_idx : part_idx + len(marker)] == marker:
            relocated_path = REPOSITORY_ROOT.joinpath(
                *parts[part_idx:]
            ).resolve()
            if relocated_path.exists():
                return relocated_path

    return resolved_path


def load_wp4_checkpoint_path(rmb_path: Path) -> Path:
    with RmbData(str(rmb_path)) as rmb_data:
        assert DataKey.MATERIAL_PROPERTY in rmb_data, rmb_path

        if "online_pb_wp4_checkpoint" in rmb_data.attrs:
            checkpoint = rmb_data.attrs["online_pb_wp4_checkpoint"]
        else:
            attrs = rmb_data[DataKey.MATERIAL_PROPERTY].attrs
            assert "source_checkpoint" in attrs, rmb_path
            checkpoint = attrs["source_checkpoint"]

    checkpoint_path = resolve_recorded_path(checkpoint)

    assert checkpoint_path.is_file(), checkpoint_path
    assert checkpoint_path.name == "policy_best.ckpt", (
        "WrenchPredictor4 visualizations must use policy_best.ckpt",
        checkpoint_path,
    )

    return checkpoint_path


def infer_camera_name(image_feature_key: str, latent_dim: int) -> str:
    prefix = "image_vae_"
    suffix = f"_{latent_dim}"

    assert image_feature_key.startswith(prefix), image_feature_key
    assert image_feature_key.endswith(suffix), image_feature_key

    camera_name = image_feature_key[len(prefix) : -len(suffix)]

    assert camera_name, (
        "Could not infer the camera from the image-feature key; use --camera_name",
        image_feature_key,
    )

    return camera_name


def load_image_source_info(
    rmb_path: Path,
    image_feature_key: str,
    latent_dim: int,
    vae_checkpoint_override: Path | None,
    camera_name_override: str | None,
) -> tuple[Path, str, bool]:
    with RmbData(str(rmb_path)) as rmb_data:
        has_stored_features = image_feature_key in rmb_data
        feature_attrs = (
            rmb_data[image_feature_key].attrs
            if has_stored_features
            else {}
        )

        if vae_checkpoint_override is not None:
            vae_checkpoint = vae_checkpoint_override.resolve()

        elif "online_pb_image_vae_checkpoint" in rmb_data.attrs:
            vae_checkpoint = resolve_recorded_path(
                rmb_data.attrs["online_pb_image_vae_checkpoint"]
            )

        else:
            assert "model" in feature_attrs, (
                "ImageVAE checkpoint is absent from RMB metadata; "
                "use --vae_checkpoint",
                rmb_path,
            )
            vae_checkpoint = resolve_recorded_path(
                feature_attrs["model"]
            )

        if camera_name_override is not None:
            camera_name = camera_name_override

        elif "online_pb_image_vae_camera_name" in rmb_data.attrs:
            camera_name = str(
                rmb_data.attrs["online_pb_image_vae_camera_name"]
            )

        elif "source_camera" in feature_attrs:
            camera_name = str(feature_attrs["source_camera"])

        elif "source_image_key" in feature_attrs:
            source_image_key = str(feature_attrs["source_image_key"])
            suffix = "_rgb_image"

            assert source_image_key.endswith(suffix), source_image_key

            camera_name = source_image_key[: -len(suffix)]

        else:
            camera_name = infer_camera_name(
                image_feature_key,
                latent_dim,
            )

    assert vae_checkpoint.is_dir(), vae_checkpoint

    video_path = rmb_path / f"{camera_name}_rgb_image.rmb.mp4"

    assert video_path.is_file(), video_path

    return vae_checkpoint, camera_name, has_stored_features


def get_prediction_indexes(
    num_raw_steps: int,
    model_meta_info: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data_info = model_meta_info["data"]

    skip = data_info["skip"]
    horizon = data_info["horizon"]
    n_obs_steps = data_info["n_obs_steps"]
    n_action_steps = data_info["n_action_steps"]

    episode_len = len(np.arange(num_raw_steps)[::skip])

    start_idxes = np.arange(
        -(n_obs_steps - 1),
        episode_len - (horizon - 1) + (n_action_steps - 1),
    )

    assert len(start_idxes) > 0, (
        num_raw_steps,
        data_info,
    )

    latest_observation_idxes = np.clip(
        start_idxes + n_obs_steps - 1,
        0,
        episode_len - 1,
    )

    target_idxes = np.clip(
        start_idxes + horizon - 1,
        0,
        episode_len - 1,
    )

    assert len(target_idxes) == len(np.unique(target_idxes)), target_idxes

    return (
        start_idxes,
        latest_observation_idxes,
        target_idxes * skip,
    )


def encode_image_batch(
    vae,
    bgr_images: list[np.ndarray],
    image_size: tuple[int, int],
    device: torch.device,
) -> np.ndarray:
    rgb_images = np.stack(
        [
            cv2.cvtColor(
                cv2.resize(
                    image,
                    image_size,
                    interpolation=cv2.INTER_LINEAR,
                ),
                cv2.COLOR_BGR2RGB,
            )
            for image in bgr_images
        ]
    )

    images = torch.from_numpy(rgb_images).to(device)
    images = images.permute(0, 3, 1, 2).float() / 255.0

    with torch.inference_mode():
        features = vae.encoder(images).embedding

    return features.cpu().numpy()


def read_video(
    video_path: Path,
    target_raw_idxes: np.ndarray,
    expected_num_frames: int,
    vae,
    image_size: tuple[int, int],
    device: torch.device,
    encode_features: bool,
) -> tuple[list[np.ndarray], np.ndarray | None, float]:
    capture = cv2.VideoCapture(str(video_path))

    assert capture.isOpened(), video_path

    fps = capture.get(cv2.CAP_PROP_FPS)

    assert fps > 0.0, (
        video_path,
        fps,
    )

    selected_frames = []
    encoded_batches = []
    encode_batch = []

    next_target_offset = 0
    frame_idx = 0

    try:
        while True:
            success, frame = capture.read()

            if not success:
                break

            if (
                next_target_offset < len(target_raw_idxes)
                and frame_idx == target_raw_idxes[next_target_offset]
            ):
                selected_frames.append(frame.copy())
                next_target_offset += 1

            if encode_features:
                encode_batch.append(frame)

                if len(encode_batch) == BATCH_SIZE:
                    encoded_batches.append(
                        encode_image_batch(
                            vae,
                            encode_batch,
                            image_size,
                            device,
                        )
                    )
                    encode_batch = []

            frame_idx += 1

        if encode_features and encode_batch:
            encoded_batches.append(
                encode_image_batch(
                    vae,
                    encode_batch,
                    image_size,
                    device,
                )
            )

    finally:
        capture.release()

    assert frame_idx == expected_num_frames, (
        video_path,
        frame_idx,
        expected_num_frames,
    )

    assert len(selected_frames) == len(target_raw_idxes), (
        video_path,
        len(selected_frames),
        len(target_raw_idxes),
    )

    encoded_features = (
        np.concatenate(encoded_batches)
        if encode_features
        else None
    )

    return selected_frames, encoded_features, fps


def load_prediction_sequences(
    rmb_path: Path,
    model_meta_info: dict,
    encoded_image_features: np.ndarray | None,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    data_info = model_meta_info["data"]

    skip = data_info["skip"]
    image_feature_key = data_info["image_feature_key"]

    with RmbData(str(rmb_path)) as rmb_data:
        num_raw_steps = rmb_data[DataKey.TIME].shape[0]

        if encoded_image_features is None:
            image_feature = rmb_data[image_feature_key][:]
        else:
            assert encoded_image_features.shape[0] == num_raw_steps, (
                encoded_image_features.shape,
                num_raw_steps,
            )
            image_feature = encoded_image_features

        image_feature = convert_data_to_policy(
            get_skipped_data_seq(
                image_feature,
                image_feature_key,
                skip,
            ),
            image_feature_key,
        )

        state_keys = model_meta_info["state"]["keys"]

        if state_keys:
            state = np.concatenate(
                [
                    convert_data_to_policy(
                        get_skipped_data_seq(
                            rmb_data[key][:],
                            key,
                            skip,
                        ),
                        key,
                    )
                    for key in state_keys
                ],
                axis=1,
            )
        else:
            state = np.zeros(
                (len(image_feature), 0),
                dtype=np.float32,
            )

        action = np.concatenate(
            [
                convert_data_to_policy(
                    get_skipped_data_seq(
                        rmb_data[key][:],
                        key,
                        skip,
                    ),
                    key,
                )
                for key in model_meta_info["action"]["keys"]
            ],
            axis=1,
        )

        online_pb = get_skipped_data_seq(
            rmb_data[DataKey.MATERIAL_PROPERTY][:],
            DataKey.MATERIAL_PROPERTY,
            skip,
        )

        raw_time = rmb_data[DataKey.TIME][:]

    episode_len = len(raw_time[::skip])

    assert state.shape[0] == episode_len, state.shape
    assert action.shape[0] == episode_len, action.shape
    assert image_feature.shape[0] == episode_len, image_feature.shape
    assert online_pb.shape[0] == episode_len, online_pb.shape

    sequences = {
        "state": normalize_data(
            state,
            model_meta_info["state"],
        ),
        "action": normalize_data(
            action,
            model_meta_info["action"],
        ),
        "image_feature": normalize_data(
            image_feature,
            model_meta_info["image_feature"],
        ),
    }

    return (
        sequences,
        online_pb.astype(np.float32),
        raw_time,
    )


def predict_farthest_features(
    policy,
    model_meta_info: dict,
    sequences: dict[str, np.ndarray],
    online_pb: np.ndarray,
    reference_pbs: np.ndarray,
    start_idxes: np.ndarray,
    latest_observation_idxes: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    horizon = model_meta_info["data"]["horizon"]
    episode_len = sequences["image_feature"].shape[0]

    predicted_batches = []
    online_pb_batches = []

    for batch_start in range(
        0,
        len(start_idxes),
        BATCH_SIZE,
    ):
        batch_start_idxes = start_idxes[
            batch_start : batch_start + BATCH_SIZE
        ]

        batch_observation_idxes = latest_observation_idxes[
            batch_start : batch_start + BATCH_SIZE
        ]

        time_idxes = np.clip(
            batch_start_idxes[:, np.newaxis]
            + np.arange(horizon),
            0,
            episode_len - 1,
        )

        batch = {
            key: torch.tensor(
                value[time_idxes],
                dtype=torch.float32,
                device=device,
            )
            for key, value in sequences.items()
        }

        batch_online_pb = online_pb[batch_observation_idxes]

        material_pbs = np.concatenate(
            [
                batch_online_pb[:, np.newaxis],
                np.broadcast_to(
                    reference_pbs[np.newaxis],
                    (
                        len(batch_start_idxes),
                        *reference_pbs.shape,
                    ),
                ),
            ],
            axis=1,
        )

        num_conditions = material_pbs.shape[1]

        repeated_batch = {
            key: value.repeat_interleave(
                num_conditions,
                dim=0,
            )
            for key, value in batch.items()
        }

        material_pbs_tensor = torch.tensor(
            material_pbs.reshape(
                -1,
                material_pbs.shape[-1],
            ),
            dtype=torch.float32,
            device=device,
        )

        with torch.inference_mode():
            prediction = policy(
                repeated_batch,
                material_pbs_tensor,
            )["image_feature"]

        farthest_prediction = prediction[:, -1].reshape(
            len(batch_start_idxes),
            num_conditions,
            -1,
        )

        predicted_batches.append(
            farthest_prediction.cpu().numpy()
        )

        online_pb_batches.append(batch_online_pb)

    normalized_features = np.concatenate(predicted_batches)

    predicted_features = denormalize_data(
        normalized_features,
        model_meta_info["image_feature"],
    )

    return (
        predicted_features,
        np.concatenate(online_pb_batches),
    )


def decode_image_features(
    vae,
    features: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    flat_features = features.reshape(
        -1,
        features.shape[-1],
    )

    reconstructed_batches = []

    for start in range(
        0,
        len(flat_features),
        BATCH_SIZE,
    ):
        latent = torch.tensor(
            flat_features[start : start + BATCH_SIZE],
            dtype=torch.float32,
            device=device,
        )

        with torch.inference_mode():
            reconstruction = vae.decoder(
                latent
            ).reconstruction

        reconstructed_batches.append(
            reconstruction.permute(
                0,
                2,
                3,
                1,
            ).cpu().numpy()
        )

    reconstructed = np.concatenate(
        reconstructed_batches
    )

    reconstructed = (
        np.round(255.0 * reconstructed)
        .clip(0, 255)
        .astype(np.uint8)
    )

    return reconstructed.reshape(
        *features.shape[:-1],
        *reconstructed.shape[1:],
    )


def get_fitted_image_geometry(
    image: np.ndarray,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    image_height, image_width = image.shape[:2]

    scale = min(
        width / image_width,
        height / image_height,
    )

    resized_width = max(
        1,
        int(round(image_width * scale)),
    )

    resized_height = max(
        1,
        int(round(image_height * scale)),
    )

    x = (width - resized_width) // 2
    y = (height - resized_height) // 2

    return (
        resized_width,
        resized_height,
        x,
        y,
    )


def fit_image(
    image: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    resized_width, resized_height, x, y = (
        get_fitted_image_geometry(
            image,
            width,
            height,
        )
    )

    resized = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=(
            cv2.INTER_AREA
            if resized_width < image.shape[1]
            else cv2.INTER_LINEAR
        ),
    )

    panel = np.zeros(
        (height, width, 3),
        dtype=np.uint8,
    )

    panel[
        y : y + resized_height,
        x : x + resized_width
    ] = resized

    return panel


# ============================================================
# Captionを動画パネルの下側に表示する関数
# ============================================================
def add_label(
    image: np.ndarray,
    text: str,
    x: int,
    y: int,
    panel_width: int | None = None,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.46
    thickness = 1

    # Captionの大きさを取得
    (text_width, text_height), baseline = cv2.getTextSize(
        text,
        font,
        font_scale,
        thickness,
    )

    # panel_widthが指定されていれば、
    # Captionをパネルの中央寄りに配置する
    if panel_width is not None:
        text_x = x + max(
            5,
            (panel_width - text_width) // 2,
        )
    else:
        text_x = x + 7

    # y は「Captionを表示する場所の下端」
    text_y = y

    # 黒い縁取り
    cv2.putText(
        image,
        text,
        (text_x, text_y),
        font,
        font_scale,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )

    # 白い文字
    cv2.putText(
        image,
        text,
        (text_x, text_y),
        font,
        font_scale,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def format_pb(pb: np.ndarray) -> str:
    return "[" + ", ".join(
        f"{value:.4f}" for value in pb
    ) + "]"


def make_canvas(
    original_frame: np.ndarray,
    reconstructed_rgb: np.ndarray,
    online_pb: np.ndarray,
    reference_pbs: np.ndarray,
    camera_name: str,
    horizon: int,
    target_time: float,
    output_size: tuple[int, int],
) -> np.ndarray:
    output_width, output_height = output_size

    top_height = output_height // 2
    top_width = output_width // 2

    canvas = np.zeros(
        (output_height, output_width, 3),
        dtype=np.uint8,
    )

    # ========================================================
    # 上段
    # ========================================================

    original_panel = fit_image(
        original_frame,
        top_width,
        top_height,
    )

    online_bgr = cv2.cvtColor(
        reconstructed_rgb[0],
        cv2.COLOR_RGB2BGR,
    )

    online_panel = fit_image(
        online_bgr,
        output_width - top_width,
        top_height,
    )

    canvas[
        :top_height,
        :top_width
    ] = original_panel

    canvas[
        :top_height,
        top_width:
    ] = online_panel

    # --------------------------------------------------------
    # OriginalのCaption
    # 「上」ではなく「左上パネルの下端」に表示
    # --------------------------------------------------------
    add_label(
        canvas,
        f"Original {camera_name}  t={target_time:.2f}s",
        0,
        top_height - 8,
        panel_width=top_width,
    )

    # --------------------------------------------------------
    # Online PBのCaption
    # 「上」ではなく「右上パネルの下端」に表示
    # --------------------------------------------------------
    online_panel_width = output_width - top_width

    add_label(
        canvas,
        f"Online PB={format_pb(online_pb)}  horizon={horizon}",
        top_width,
        top_height - 8,
        panel_width=online_panel_width,
    )

    # ========================================================
    # 下段
    # ========================================================

    bottom_edges = np.rint(
        np.linspace(
            0,
            output_width,
            4,
        )
    ).astype(int)

    for panel_idx, object_id in enumerate(
        REFERENCE_OBJECT_IDS
    ):
        x0 = bottom_edges[panel_idx]
        x1 = bottom_edges[panel_idx + 1]

        reconstructed_bgr = cv2.cvtColor(
            reconstructed_rgb[panel_idx + 1],
            cv2.COLOR_RGB2BGR,
        )

        panel_width = x1 - x0
        panel_height = output_height - top_height

        panel = fit_image(
            reconstructed_bgr,
            panel_width,
            panel_height,
        )

        canvas[
            top_height:,
            x0:x1
        ] = panel

        # ----------------------------------------------------
        # Captionを画像の「下側」に表示
        # ----------------------------------------------------
        add_label(
            canvas,
            f"PB{object_id}={format_pb(reference_pbs[panel_idx])}",
            x0,
            output_height - 8,
            panel_width=panel_width,
        )

    return canvas


def encode_video(
    output_path: Path,
    original_frames: list[np.ndarray],
    reconstructed: np.ndarray,
    online_pbs: np.ndarray,
    reference_pbs: np.ndarray,
    camera_name: str,
    horizon: int,
    target_times: np.ndarray,
    fps: float,
    output_size: tuple[int, int],
) -> None:
    assert output_path.suffix == ".mp4", output_path

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    work_path = output_path.with_suffix(".mp4v.mp4")
    encoded_path = output_path.with_suffix(".h264.mp4")

    writer = cv2.VideoWriter(
        str(work_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        output_size,
    )

    assert writer.isOpened(), work_path

    try:
        for frame_idx, original_frame in enumerate(
            original_frames
        ):
            canvas = make_canvas(
                original_frame,
                reconstructed[frame_idx],
                online_pbs[frame_idx],
                reference_pbs,
                camera_name,
                horizon,
                target_times[frame_idx],
                output_size,
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


def get_output_path(
    rmb_path: Path,
    input_path: Path,
    output: Path | None,
    output_dir: Path | None,
    camera_name: str,
) -> Path:
    if output is not None:
        return output

    if output_dir is None:
        if input_path.suffix == ".rmb":
            output_dir = rmb_path.parent
        else:
            output_dir = input_path.with_name(
                f"{input_path.name}_OnlinePbReconstructionVideos"
            )

    return (
        output_dir
        / f"{rmb_path.stem}_{camera_name}_online_pb_reconstruction.mp4"
    )


def main() -> None:
    args = parse_args()

    assert args.rmb_path.is_dir(), args.rmb_path

    assert (
        args.output_width > 0
        and args.output_width % 2 == 0
    ), args.output_width

    assert (
        args.output_height > 0
        and args.output_height % 2 == 0
    ), args.output_height

    rmb_paths = sorted(
        Path(path)
        for path in find_rmb_files(
            str(args.rmb_path)
        )
    )

    assert rmb_paths, args.rmb_path

    assert all(
        path.suffix == ".rmb"
        for path in rmb_paths
    ), rmb_paths

    if len(rmb_paths) > 1:
        assert (
            args.output is None
        ), "Use --output_dir for multiple RMB episodes."
    else:
        assert (
            args.output_dir is None
            or args.output is None
        )

    checkpoint_paths = [
        load_wp4_checkpoint_path(path)
        for path in rmb_paths
    ]

    checkpoint_path = checkpoint_paths[0]

    assert all(
        path == checkpoint_path
        for path in checkpoint_paths
    ), checkpoint_paths

    model_meta_info = load_model_meta_info(
        checkpoint_path
    )

    assert (
        model_meta_info["data"]["horizon"] == 16
    ), model_meta_info["data"]

    image_feature_key = model_meta_info["data"][
        "image_feature_key"
    ]

    assert image_feature_key.startswith(
        "image_vae"
    ), (
        "This video requires an ImageVAE-based WP4",
        image_feature_key,
    )

    latent_dim = model_meta_info["policy"]["args"][
        "image_feature_dim"
    ]

    image_source_info = [
        load_image_source_info(
            path,
            image_feature_key,
            latent_dim,
            args.vae_checkpoint,
            args.camera_name,
        )
        for path in rmb_paths
    ]

    vae_checkpoint, camera_name, _ = (
        image_source_info[0]
    )

    assert all(
        info[:2] == (vae_checkpoint, camera_name)
        for info in image_source_info
    ), image_source_info

    assert torch.cuda.is_available(), (
        "WP4 reconstruction requires a CUDA GPU."
    )

    device = torch.device("cuda")

    policy = load_policy(
        checkpoint_path,
        model_meta_info,
        device,
    )

    reference_pb_table, _ = load_pb_table(
        checkpoint_path,
        model_meta_info,
    )

    reference_pbs = reference_pb_table[
        list(REFERENCE_OBJECT_IDS)
    ]

    vae = (
        AutoModel.load_from_folder(
            str(vae_checkpoint)
        )
        .eval()
        .to(device)
    )

    vae.requires_grad_(False)

    assert (
        vae.model_config.latent_dim == latent_dim
    ), (
        vae.model_config.latent_dim,
        latent_dim,
    )

    _, image_height, image_width = (
        vae.model_config.input_dim
    )

    image_size = (
        image_width,
        image_height,
    )

    output_size = (
        args.output_width,
        args.output_height,
    )

    print(
        f"WP4 checkpoint: {checkpoint_path}"
    )
    print(
        f"ImageVAE checkpoint: {vae_checkpoint}"
    )
    print(
        f"Image feature: "
        f"{image_feature_key} ({latent_dim}D)"
    )
    print(
        f"Camera: {camera_name}"
    )
    print(
        f"Reference PBs: "
        f"{reference_pbs.tolist()}"
    )
    print(
        f"Output size: "
        f"{output_size[0]}x{output_size[1]}"
    )
    print(
        f"Episodes: {len(rmb_paths)}"
    )

    for episode_idx, (
        rmb_path,
        source_info,
    ) in enumerate(
        zip(
            rmb_paths,
            image_source_info,
            strict=True,
        ),
        start=1,
    ):
        _, _, has_stored_features = source_info

        with RmbData(str(rmb_path)) as rmb_data:
            num_raw_steps = rmb_data[
                DataKey.TIME
            ].shape[0]

        (
            start_idxes,
            latest_observation_idxes,
            target_raw_idxes,
        ) = get_prediction_indexes(
            num_raw_steps,
            model_meta_info,
        )

        video_path = (
            rmb_path
            / f"{camera_name}_rgb_image.rmb.mp4"
        )

        (
            original_frames,
            encoded_features,
            source_fps,
        ) = read_video(
            video_path,
            target_raw_idxes,
            num_raw_steps,
            vae,
            image_size,
            device,
            encode_features=not has_stored_features,
        )

        (
            sequences,
            online_pb,
            raw_time,
        ) = load_prediction_sequences(
            rmb_path,
            model_meta_info,
            encoded_features,
        )

        (
            predicted_features,
            prediction_online_pbs,
        ) = predict_farthest_features(
            policy,
            model_meta_info,
            sequences,
            online_pb,
            reference_pbs,
            start_idxes,
            latest_observation_idxes,
            device,
        )

        reconstructed = decode_image_features(
            vae,
            predicted_features,
            device,
        )

        target_times = (
            raw_time[target_raw_idxes]
            - raw_time[0]
        )

        output_path = get_output_path(
            rmb_path,
            args.rmb_path,
            args.output,
            args.output_dir,
            camera_name,
        )

        encode_video(
            output_path,
            original_frames,
            reconstructed,
            prediction_online_pbs,
            reference_pbs,
            camera_name,
            model_meta_info["data"]["horizon"],
            target_times,
            source_fps / model_meta_info["data"]["skip"],
            output_size,
        )

        print(
            f"[{episode_idx}/{len(rmb_paths)}] "
            f"{rmb_path.name}: "
            f"{len(original_frames)} frames -> "
            f"{output_path.resolve()}"
        )


if __name__ == "__main__":
    main()