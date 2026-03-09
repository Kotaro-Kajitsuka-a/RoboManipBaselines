import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    cwd = Path(".").resolve()
    if cwd.name != "arucoboard":
        raise RuntimeError("Run from the 'arucoboard' directory.")
    parser = argparse.ArgumentParser(description="Train YOLO pose model.")
    parser.add_argument("data", help="Path to dataset YAML.")
    args = parser.parse_args()

    data_path = Path(args.data).expanduser().resolve()
    run_name = data_path.parent.name

    # Load a model
    model = YOLO("yolopose/yolo26n-pose.pt")  # load a pretrained model (recommended for training)
    # Train the model
    model.train(
        data=str(data_path),
        epochs=100,
        imgsz=640,
        name=run_name,
        plots=True
    )


if __name__ == "__main__":
    main()
