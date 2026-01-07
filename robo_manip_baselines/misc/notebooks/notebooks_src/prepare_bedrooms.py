from pathlib import Path
import subprocess


TARGET_MP4_NAME = "front_rgb_image.rmb.mp4"


def iter_target_mp4(input_dir: Path):
    for mp4_path in input_dir.rglob(TARGET_MP4_NAME):
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


def prepare_bedrooms(input_dir: Path):
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"input_dir does not exist: {input_dir}")

    mp4_paths = list(iter_target_mp4(input_dir))
    print(f"[prepare_bedrooms] Found mp4 files: {len(mp4_paths)}")
    for mp4_path in mp4_paths:
        extract_frames(mp4_path, bedroom_dir_for_mp4(mp4_path))
    return mp4_paths


def list_bedroom_dirs(input_dir: Path):
    input_dir = Path(input_dir)
    bedroom_dirs = sorted(input_dir.rglob(".bedrooms/bedroom_front_rgb_image"))
    return [str(p) for p in bedroom_dirs]
