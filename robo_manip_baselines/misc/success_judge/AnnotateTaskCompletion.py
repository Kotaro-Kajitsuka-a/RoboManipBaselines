import argparse
import csv
import os
from pathlib import Path

import cv2
import numpy as np

from robo_manip_baselines.common import DataKey, RmbData, find_rmb_files

CSV_COLUMNS = ["rmb_path", "camera", "completion_frame", "completion_time", "success"]


def parse_argument():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "path",
        type=str,
        help="path to data (*.rmb or *.hdf5) or directory containing them",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default=None,
        help="camera name to annotate. If omitted, the first rgb image key is used.",
    )
    parser.add_argument(
        "--csv_name",
        type=str,
        default="task_completion.csv",
        help="output csv filename. It is created under the given data path directory.",
    )
    parser.add_argument(
        "--redo",
        action="store_true",
        help="annotate files again even if they already exist in the csv.",
    )
    return parser.parse_args()


def get_output_csv_path(path, csv_name):
    path = Path(path).expanduser()
    if path.is_dir():
        return path / csv_name
    return path.parent / csv_name


def load_done_paths(csv_path):
    if not csv_path.exists():
        return set()

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        assert (
            reader.fieldnames == CSV_COLUMNS
        ), f"Unexpected csv columns: {reader.fieldnames}. Expected: {CSV_COLUMNS}"
        return {row["rmb_path"] for row in reader}


def append_result(csv_path, result):
    file_exists = csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)


def to_csv_rmb_path(rmb_path, csv_path):
    return os.path.relpath(rmb_path, start=csv_path.parent)


def normalize_done_path(saved_rmb_path, csv_path):
    path = Path(saved_rmb_path).expanduser()
    if not path.is_absolute():
        path = csv_path.parent / path
    return str(path.resolve())


def find_rgb_key(rmb_data, camera):
    if camera is not None:
        rgb_key = DataKey.get_rgb_image_key(camera)
        if rgb_key not in rmb_data:
            raise KeyError(f"RGB image key not found: {rgb_key}")
        return rgb_key

    rgb_keys = sorted([key for key in rmb_data.keys() if DataKey.is_rgb_image_key(key)])
    if len(rgb_keys) == 0:
        raise KeyError("RGB image key not found.")
    return rgb_keys[0]


class RmbFrameReader:
    def __init__(self, rmb_path, rgb_key):
        self.rmb_path = rmb_path
        self.rgb_key = rgb_key
        self.rmb_data = RmbData(rmb_path)
        self.rmb_data.open()
        self.time_seq = np.asarray(
            self.rmb_data[DataKey.TIME][:], dtype=np.float64
        ).reshape(-1)
        assert len(self.time_seq) > 0

        self.cap = None
        mp4_path = Path(rmb_path) / f"{rgb_key}.rmb.mp4"
        if Path(rmb_path).suffix == ".rmb" and mp4_path.exists():
            self.cap = cv2.VideoCapture(str(mp4_path))
            if not self.cap.isOpened():
                self.cap.release()
                self.cap = None

        if self.cap is None:
            self.rgb_seq = self.rmb_data[rgb_key]
            self.length = int(self.rgb_seq.shape[0])
        else:
            self.rgb_seq = None
            self.length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        assert self.length == len(
            self.time_seq
        ), f"frame length mismatch: {self.length} != {len(self.time_seq)}"
        self._last_idx = None
        self._last_frame = None

    def close(self):
        if self.cap is not None:
            self.cap.release()
        self.rmb_data.close()

    def get_frame(self, idx):
        idx = int(idx)
        assert 0 <= idx < self.length
        if self._last_idx == idx:
            return self._last_frame

        if self.cap is None:
            rgb = self.rgb_seq[idx]
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, bgr = self.cap.read()
            if not ok:
                raise RuntimeError(f"Failed to read frame {idx}: {self.rmb_path}")

        self._last_idx = idx
        self._last_frame = bgr
        return bgr


class TaskCompletionAnnotator:
    def __init__(self, path, camera, csv_name, redo):
        self.path = path
        self.camera = camera
        self.csv_path = get_output_csv_path(path, csv_name)
        self.redo = redo

    def run(self):
        rmb_path_list = find_rmb_files(self.path)
        done_paths = set()
        if not self.redo:
            done_paths = {
                normalize_done_path(path, self.csv_path)
                for path in load_done_paths(self.csv_path)
            }

        print(f"[{self.__class__.__name__}] Output csv: {self.csv_path}")
        print(f"[{self.__class__.__name__}] Found {len(rmb_path_list)} files.")

        for file_idx, rmb_path in enumerate(rmb_path_list):
            abs_rmb_path = str(Path(rmb_path).expanduser().resolve())
            if abs_rmb_path in done_paths:
                print(f"[{self.__class__.__name__}] Skip annotated: {abs_rmb_path}")
                continue

            result = self.annotate_one(abs_rmb_path, file_idx, len(rmb_path_list))
            if result is None:
                print(f"[{self.__class__.__name__}] Quit.")
                break

            result["rmb_path"] = to_csv_rmb_path(abs_rmb_path, self.csv_path)
            append_result(self.csv_path, result)
            done_paths.add(abs_rmb_path)
            print(f"[{self.__class__.__name__}] Saved: {result}")

    def annotate_one(self, rmb_path, file_idx, num_files):
        with RmbData(rmb_path) as rmb_data:
            rgb_key = find_rgb_key(rmb_data, self.camera)
        camera = DataKey.get_camera_name(rgb_key)

        reader = RmbFrameReader(rmb_path, rgb_key)
        window_name = "AnnotateTaskCompletion"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        frame_idx = 0
        try:
            while True:
                bgr = reader.get_frame(frame_idx).copy()
                self.draw_overlay(
                    bgr=bgr,
                    rmb_path=rmb_path,
                    camera=camera,
                    file_idx=file_idx,
                    num_files=num_files,
                    frame_idx=frame_idx,
                    num_frames=reader.length,
                    time_value=reader.time_seq[frame_idx],
                )
                cv2.imshow(window_name, bgr)
                key = cv2.waitKey(0) & 0xFF

                if key in (ord("d"), 83):
                    frame_idx = min(frame_idx + 1, reader.length - 1)
                elif key in (ord("a"), 81):
                    frame_idx = max(frame_idx - 1, 0)
                elif key == ord("l"):
                    frame_idx = min(frame_idx + 10, reader.length - 1)
                elif key == ord("j"):
                    frame_idx = max(frame_idx - 10, 0)
                elif key == ord("g"):
                    frame_idx = 0
                elif key == ord("G"):
                    frame_idx = reader.length - 1
                elif key == ord(" "):
                    return {
                        "rmb_path": rmb_path,
                        "camera": camera,
                        "completion_frame": frame_idx,
                        "completion_time": f"{reader.time_seq[frame_idx]:.9f}",
                        "success": 1,
                    }
                elif key == ord("f"):
                    return {
                        "rmb_path": rmb_path,
                        "camera": camera,
                        "completion_frame": -1,
                        "completion_time": "",
                        "success": 0,
                    }
                elif key == ord("q"):
                    return None
        finally:
            reader.close()
            cv2.destroyWindow(window_name)

    @staticmethod
    def draw_overlay(
        bgr,
        rmb_path,
        camera,
        file_idx,
        num_files,
        frame_idx,
        num_frames,
        time_value,
    ):
        lines = [
            f"{file_idx + 1}/{num_files}  camera={camera}",
            f"frame={frame_idx}/{num_frames - 1}  time={time_value:.3f}s",
            "a/d: +/-1  j/l: +/-10  g/G: first/last",
            "space: success at this frame  f: failure  q: quit",
            os.path.basename(rmb_path),
        ]

        x, y0, dy = 12, 24, 24
        width = max(680, min(bgr.shape[1], 1050))
        height = y0 + dy * len(lines)
        cv2.rectangle(bgr, (0, 0), (width, height), (0, 0, 0), -1)
        for i, line in enumerate(lines):
            y = y0 + dy * i
            cv2.putText(
                bgr,
                line,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )


if __name__ == "__main__":
    annotator = TaskCompletionAnnotator(**vars(parse_argument()))
    annotator.run()
