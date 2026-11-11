import argparse

import cv2
import numpy as np

from robo_manip_baselines.common import DataKey, DataManager


def make_montage(images, labels, tile_size=(640, 480)):
    tile_width, tile_height = tile_size
    label_height = 32
    column_count = min(2, len(images))
    row_count = (len(images) + column_count - 1) // column_count
    montage = np.zeros(
        (row_count * (tile_height + label_height), column_count * tile_width, 3),
        dtype=np.uint8,
    )

    for image_idx, (image, label) in enumerate(zip(images, labels)):
        image_height, image_width = image.shape[:2]
        scale = min(tile_width / image_width, tile_height / image_height)
        resized_width = max(1, round(image_width * scale))
        resized_height = max(1, round(image_height * scale))
        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_NEAREST
        image = cv2.resize(
            image, (resized_width, resized_height), interpolation=interpolation
        )

        row_idx, column_idx = divmod(image_idx, column_count)
        x = column_idx * tile_width + (tile_width - resized_width) // 2
        y = row_idx * (tile_height + label_height) + label_height
        montage[y : y + resized_height, x : x + resized_width] = image
        cv2.putText(
            montage,
            label,
            (column_idx * tile_width + 10, row_idx * (tile_height + label_height) + 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return montage


def parse_argument():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("rmb_filename", type=str)
    parser.add_argument(
        "--pressure_sensor_key",
        type=str,
        default="main_pressure_sensor",
        help="Data key of the recorded pressure sensor.",
    )
    parser.add_argument("--skip", type=int, default=1, help="Number of frames to skip.")
    parser.add_argument("--fps", type=float, default=60, help="Playback frame rate.")
    return parser.parse_args()


def main():
    args = parse_argument()
    if args.skip < 1:
        raise ValueError("--skip must be at least 1.")
    if args.fps <= 0:
        raise ValueError("--fps must be positive.")

    data_manager = DataManager(env=None)
    data_manager.load_data(
        args.rmb_filename,
        load_keys=[args.pressure_sensor_key, DataKey.TIME],
    )
    try:
        camera_names = data_manager.get_meta_data("camera_names").tolist()
    except KeyError:
        camera_names = []
    try:
        rgb_tactile_names = data_manager.get_meta_data("rgb_tactile_names").tolist()
    except KeyError:
        try:
            rgb_tactile_names = data_manager.get_meta_data("tactile_names").tolist()
        except KeyError:
            rgb_tactile_names = []
    image_names = camera_names + rgb_tactile_names
    rgb_keys = [DataKey.get_rgb_image_key(name) for name in image_names]
    data_manager.load_data(
        args.rmb_filename,
        load_keys=[args.pressure_sensor_key, DataKey.TIME, *rgb_keys],
    )

    if args.pressure_sensor_key not in data_manager.all_data_seq:
        raise KeyError(
            f"Pressure sensor key '{args.pressure_sensor_key}' was not found in the RMB data."
        )

    matrices = np.asarray(data_manager.get_data_seq(args.pressure_sensor_key))
    if matrices.ndim != 3 or len(matrices) == 0:
        raise ValueError(
            f"Expected '{args.pressure_sensor_key}' to contain a non-empty sequence of 2D matrices."
        )

    rgb_images = {}
    for name, key in zip(image_names, rgb_keys):
        if key not in data_manager.all_data_seq:
            raise KeyError(f"RGB image key '{key}' was not found in the RMB data.")
        images = np.asarray(data_manager.get_data_seq(key))
        if images.ndim != 4 or images.shape[-1] != 3:
            raise ValueError(f"Expected '{key}' to contain a sequence of RGB images.")
        if len(images) != len(matrices):
            raise ValueError(
                f"RGB image key '{key}' has {len(images)} frames, but the pressure sensor has {len(matrices)}."
            )
        rgb_images[name] = images

    times = np.asarray(data_manager.get_data_seq(DataKey.TIME))
    if len(times) != len(matrices):
        times = None

    window_name = "RGB and pressure sensor"
    frame_idx = 0
    is_playing = True
    wait_ms = max(1, round(1000 / args.fps))
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    print("Press Space to pause/resume, A/D to step, or Q/Esc to quit.")
    try:
        while frame_idx < len(matrices):
            title = f"Frame {frame_idx + 1}/{len(matrices)}"
            if times is not None:
                title += f" ({times[frame_idx]:.2f} s)"

            display_images = []
            display_labels = []
            for name, images in rgb_images.items():
                image = cv2.cvtColor(images[frame_idx], cv2.COLOR_RGB2BGR)
                display_images.append(image)
                display_labels.append(f"{name} RGB")

            image = np.clip(matrices[frame_idx][::-1, :], 0, 255).astype(np.uint8)
            image = cv2.applyColorMap(image, cv2.COLORMAP_TURBO)
            display_images.append(image)
            display_labels.append(f"Pressure sensor - {title}")
            cv2.imshow(window_name, make_montage(display_images, display_labels))

            key = cv2.waitKey(wait_ms if is_playing else 30) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                is_playing = not is_playing
            elif key == ord("a"):
                is_playing = False
                frame_idx = max(0, frame_idx - args.skip)
            elif key == ord("d"):
                is_playing = False
                frame_idx = min(len(matrices) - 1, frame_idx + args.skip)
            elif is_playing:
                frame_idx += args.skip
    finally:
        cv2.destroyWindow(window_name)


if __name__ == "__main__":
    main()
