import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    cwd = Path(".").resolve()
    if cwd.name != "arucoboard":
        raise RuntimeError("Run from the 'arucoboard' directory.")

    parser = argparse.ArgumentParser(description="Track with a YOLO pose model.")
    parser.add_argument("pt_path", help="Path to .pt model file.")
    parser.add_argument("mp4_path", help="Path to mp4 file.")
    args = parser.parse_args()

    model = YOLO(args.pt_path)
    model.track(args.mp4_path, show=True, tracker="bytetrack.yaml")


if __name__ == "__main__":
    main()
