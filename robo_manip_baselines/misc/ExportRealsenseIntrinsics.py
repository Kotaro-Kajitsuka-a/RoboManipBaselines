import argparse
from datetime import datetime, timezone
from pathlib import Path

import yaml


def _load_camera_ids(config_path: Path) -> dict:
    with config_path.open("r") as f:
        data = yaml.safe_load(f)
    camera_ids = data.get("camera_ids")
    if not isinstance(camera_ids, dict) or not camera_ids:
        raise ValueError(f"camera_ids not found in: {config_path}")
    return camera_ids


def _get_realsense_intrinsics(serial: str, width: int, height: int, fps: int) -> dict:
    try:
        import pyrealsense2 as rs  # type: ignore
    except Exception as exc:
        raise RuntimeError("pyrealsense2 is required to read RealSense intrinsics.") from exc

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    profile = pipeline.start(config)
    try:
        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_stream.get_intrinsics()
        return {
            "width": int(intr.width),
            "height": int(intr.height),
            "fx": float(intr.fx),
            "fy": float(intr.fy),
            "cx": float(intr.ppx),
            "cy": float(intr.ppy),
            "coeffs": [float(v) for v in intr.coeffs[:5]],
            "model": str(intr.model),
        }
    finally:
        pipeline.stop()


def _build_output(camera_ids: dict, width: int, height: int, fps: int) -> dict:
    output = {
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "image_size": [int(width), int(height)],
        "fps": int(fps),
        "cameras": {},
    }
    for camera_name, serial in camera_ids.items():
        if not serial:
            raise ValueError(f"Empty serial for camera '{camera_name}'.")
        intr = _get_realsense_intrinsics(serial, width, height, fps)
        output["cameras"][camera_name] = {
            "serial": str(serial),
            "intrinsics": intr,
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export RealSense color intrinsics into a YAML file."
    )
    parser.add_argument(
        "--config",
        default="robo_manip_baselines/envs/configs/RealXarm7DualDemoEnv.yaml",
        help="Path to RealXarm7DualDemoEnv.yaml",
    )
    parser.add_argument(
        "--output",
        default="robo_manip_baselines/misc/arucoboard/camera_intrinsics.yaml",
        help="Output YAML path.",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    camera_ids = _load_camera_ids(config_path)
    output = _build_output(camera_ids, args.width, args.height, args.fps)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        yaml.safe_dump(output, f, sort_keys=False)
    print(f"Saved intrinsics to: {output_path}")


if __name__ == "__main__":
    main()
