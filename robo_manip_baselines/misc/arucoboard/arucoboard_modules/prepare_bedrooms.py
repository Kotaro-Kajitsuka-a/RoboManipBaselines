from pathlib import Path
import subprocess


def _target_mp4_name(camera_name: str) -> str:
    if not camera_name:
        raise ValueError("camera_name must be a non-empty string.")
    return f"{camera_name}_rgb_image.rmb.mp4"


def iter_target_mp4(input_dir: Path, camera_name: str):
    target_name = _target_mp4_name(camera_name)
    for mp4_path in input_dir.rglob(target_name):
        if ".bedrooms" in mp4_path.parts:
            continue
        yield mp4_path


def bedroom_dir_for_mp4(mp4_path: Path) -> Path:
    base = Path(Path(mp4_path.name).stem).stem
    return mp4_path.parent / ".bedrooms" / f"bedroom_{base}"


def extract_frames(mp4_path: Path, bedroom_dir: Path):
    bedroom_dir.mkdir(parents=True, exist_ok=True)
    output_pattern = str(bedroom_dir / "%05d.jpg")
    command = [
        "ffmpeg",
        "-i",
        str(mp4_path),
        "-q:v",
        "2",
        "-start_number",
        "0",
        output_pattern,
    ]
    print(f"[prepare_bedrooms] ffmpeg: {' '.join(command)}")
    subprocess.run(command, check=True)


def prepare_bedrooms(input_dir: Path, camera_name: str):
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"input_dir does not exist: {input_dir}")

    mp4_paths = list(iter_target_mp4(input_dir, camera_name))
    print(f"[prepare_bedrooms] Found mp4 files: {len(mp4_paths)}")
    for mp4_path in mp4_paths:
        extract_frames(mp4_path, bedroom_dir_for_mp4(mp4_path))
    return mp4_paths


def list_bedroom_dirs(input_dir: Path, camera_name: str):
    input_dir = Path(input_dir)
    base = f"bedroom_{camera_name}_rgb_image"
    bedroom_dirs = sorted(input_dir.rglob(f".bedrooms/{base}"))
    return [str(p) for p in bedroom_dirs]
