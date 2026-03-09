import argparse
import shutil
from pathlib import Path

import cv2
import h5py
import numpy as np
from cv2 import aruco

from robo_manip_baselines.misc.arucoboard.arucoboard_modules.aruco_prompt import (
    ARUCO_DICT_ID,
    GRIDBOARD_H_M,
    GRIDBOARD_W_M,
    MARKERS_X,
    MARKERS_Y,
    MARKER_LENGTH_M,
    MARKER_SEPARATION_M,
    load_camera_intrinsics,
)

BOX_DEPTH_M = 0.1140
BASE_T_CAM_PATH = Path(__file__).resolve().parents[1] / "calib" / "base_center_T_top.calib"
CAMERA_INTRINSICS_PATH = Path(__file__).resolve().parent / "arucoboard" / "camera_intrinsics.yaml"
T_MAX_SECONDS = 25.0
WARNING_CONSECUTIVE_MISSING = 10
MAX_CONSECUTIVE_MISSING = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create main_eval.hdf5 from ArUco Marker Board estimation."
    )
    parser.add_argument("dataset_dir", type=str, help="Path to dataset directory")
    return parser.parse_args()


def _rotation_matrix_to_6d(rotation: np.ndarray) -> np.ndarray:
    if rotation.shape != (3, 3):
        raise ValueError(f"rotation must be 3x3, got {rotation.shape}")

    x_axis = rotation[:, 0]
    y_axis = rotation[:, 1]

    x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-8)
    y_axis = y_axis - np.dot(x_axis, y_axis) * x_axis
    y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-8)

    return np.concatenate([x_axis, y_axis]).astype(np.float32)


def _rotation_matrix_to_z_degree(rotation: np.ndarray) -> float:
    if rotation.shape != (3, 3):
        raise ValueError(f"rotation must be 3x3, got {rotation.shape}")
    z_degree_signed = float(np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0])))
    theta = (-z_degree_signed + 360.0) - 180.0
    if theta > 180.0:
        theta -= 360.0
    return theta


def _get_frame_count(video_path: Path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count


def _get_video_meta(video_path: Path) -> tuple[int, int, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid video size: {video_path}")
    if fps <= 0.0:
        fps = 30.0
    return width, height, fps


def _build_aruco_board():
    try:
        aruco_dict = aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    except AttributeError:
        aruco_dict = aruco.Dictionary_get(ARUCO_DICT_ID)

    try:
        parameters = aruco.DetectorParameters_create()
    except AttributeError:
        parameters = aruco.DetectorParameters()

    try:
        board = aruco.GridBoard(
            (MARKERS_X, MARKERS_Y), MARKER_LENGTH_M, MARKER_SEPARATION_M, aruco_dict
        )
    except Exception:
        board = aruco.GridBoard_create(
            MARKERS_X, MARKERS_Y, MARKER_LENGTH_M, MARKER_SEPARATION_M, aruco_dict
        )

    return aruco_dict, parameters, board


def _draw_axes(
    frame: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    K: np.ndarray,
    dist_coeffs: np.ndarray,
    axis_len: float = 0.05,
) -> np.ndarray:
    axes_3d = np.array(
        [
            [0.0, 0.0, 0.0],
            [axis_len, 0.0, 0.0],
            [0.0, axis_len, 0.0],
            [0.0, 0.0, axis_len],
        ],
        dtype=np.float32,
    )
    proj, _ = cv2.projectPoints(axes_3d, rvec, tvec, K, dist_coeffs)
    proj = proj.reshape(-1, 2)
    o = tuple(np.round(proj[0]).astype(int))
    x = tuple(np.round(proj[1]).astype(int))
    y = tuple(np.round(proj[2]).astype(int))
    z = tuple(np.round(proj[3]).astype(int))
    cv2.line(frame, o, x, (0, 0, 255), 2)
    cv2.line(frame, o, y, (0, 255, 0), 2)
    cv2.line(frame, o, z, (255, 0, 0), 2)
    return frame


def estimate_box_pose_sequences(
    video_path: Path,
    num_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    K, dist_coeffs = load_camera_intrinsics(CAMERA_INTRINSICS_PATH, "front")
    base_T_cam = np.loadtxt(BASE_T_CAM_PATH).astype(np.float32)
    aruco_dict, aruco_params, board = _build_aruco_board()

    width, height, fps = _get_video_meta(video_path)
    result_video_path = video_path.with_name("front_rgb_image_estimation_result_marker.rmb.mp4")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    writer = cv2.VideoWriter(
        str(result_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to create video writer: {result_video_path}")

    translation_seq = np.zeros((num_steps, 3), dtype=np.float32)
    rotation6d_seq = np.zeros((num_steps, 6), dtype=np.float32)
    z_degree_seq = np.zeros((num_steps,), dtype=np.float32)
    valid_seq = np.ones((num_steps,), dtype=np.uint8)

    R_flip_x = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32)
    center_offset = np.array([GRIDBOARD_W_M * 0.5, GRIDBOARD_H_M * 0.5, 0.0], dtype=np.float32)
    z_offset = np.array([0.0, 0.0, -BOX_DEPTH_M * 0.5], dtype=np.float32)

    consecutive_missing = 0
    last_translation = None
    last_rotation6d = None
    last_z_degree = None
    failure_message = None

    try:
        for i in range(num_steps):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Video ended early at frame {i}. expected_steps={num_steps}")
            elapsed_seconds = float(i) / float(fps)

            annotated_frame = frame.copy()
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                corners, ids, _ = aruco.detectMarkers(
                    gray, aruco_dict, parameters=aruco_params
                )
                if ids is None or len(ids) == 0:
                    raise RuntimeError("No ArUco markers detected.")

                retval, rvec, tvec = aruco.estimatePoseBoard(
                    corners, ids, board, K, dist_coeffs, None, None
                )
                if retval <= 0:
                    raise RuntimeError("aruco.estimatePoseBoard failed.")

                aruco.drawDetectedMarkers(annotated_frame, corners, ids)

                R_board, _ = cv2.Rodrigues(rvec)
                t_board_box = center_offset + R_flip_x @ z_offset
                R_cam_box = R_board @ R_flip_x
                t_cam_box = R_board @ t_board_box.reshape(3, 1) + tvec.reshape(3, 1)

                cam_T_box = np.eye(4, dtype=np.float32)
                cam_T_box[:3, :3] = R_cam_box.astype(np.float32)
                cam_T_box[:3, 3] = t_cam_box.flatten().astype(np.float32)
                base_T_box = base_T_cam @ cam_T_box

                R_base_box = base_T_box[:3, :3]
                t_base_box = base_T_box[:3, 3]
                z_degree = _rotation_matrix_to_z_degree(R_base_box)

                translation_seq[i] = t_base_box.astype(np.float32)
                rotation6d_seq[i] = _rotation_matrix_to_6d(R_base_box)
                z_degree_seq[i] = z_degree
                valid_seq[i] = 1

                last_translation = translation_seq[i].copy()
                last_rotation6d = rotation6d_seq[i].copy()
                last_z_degree = float(z_degree_seq[i])
                consecutive_missing = 0

                annotated_frame = _draw_axes(
                    annotated_frame, rvec, tvec, K, dist_coeffs, axis_len=0.06
                )
                rvec_box, _ = cv2.Rodrigues(R_cam_box)
                annotated_frame = _draw_axes(
                    annotated_frame, rvec_box, t_cam_box, K, dist_coeffs, axis_len=0.06
                )
            except Exception:
                valid_seq[i] = 0
                if elapsed_seconds <= T_MAX_SECONDS:
                    consecutive_missing += 1
                    if WARNING_CONSECUTIVE_MISSING <= consecutive_missing <= MAX_CONSECUTIVE_MISSING:
                        print(
                            "\033[1;31m"
                            f"[CreateBoxPoseEvalHdf5_Marker][SEVERE] {consecutive_missing} consecutive failures "
                            f"at frame {i} ({elapsed_seconds:.3f}s). "
                            f"Still continuing (tolerance={MAX_CONSECUTIVE_MISSING})."
                            "\033[0m"
                        )
                    if consecutive_missing > MAX_CONSECUTIVE_MISSING:
                        if failure_message is None:
                            failure_message = (
                                f"Marker detection failed for {consecutive_missing} consecutive frames "
                                f"at frame {i} ({elapsed_seconds:.3f}s). "
                                f"Exceeded tolerance={MAX_CONSECUTIVE_MISSING} within T_MAX_SECONDS={T_MAX_SECONDS:.1f}."
                            )
                else:
                    consecutive_missing = 0
                if last_translation is not None:
                    translation_seq[i] = last_translation
                    rotation6d_seq[i] = last_rotation6d
                    z_degree_seq[i] = last_z_degree

            z_text = float(z_degree_seq[i])
            cv2.putText(
                annotated_frame,
                f"frame={i:04d} z={z_text:+07.2f} deg",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            writer.write(annotated_frame)

        ok_extra, _ = cap.read()
    finally:
        cap.release()
        writer.release()

    if ok_extra:
        raise RuntimeError("Video has more frames than expected time steps.")
    if failure_message is not None:
        raise RuntimeError(failure_message)

    return translation_seq, rotation6d_seq, z_degree_seq, valid_seq


def write_eval_hdf5(
    src_hdf5_path: Path,
    translation_seq: np.ndarray,
    rotation6d_seq: np.ndarray,
    z_degree_seq: np.ndarray,
    valid_seq: np.ndarray,
) -> Path:
    dst_hdf5_path = src_hdf5_path.with_name("main_eval.hdf5")
    shutil.copy2(src_hdf5_path, dst_hdf5_path)

    with h5py.File(dst_hdf5_path, "r+") as h5file:
        for key in (
            "box_translation_base",
            "box_6d_rotation_base",
            "box_z_degree",
            "estimation_valid",
        ):
            if key in h5file:
                del h5file[key]

        h5file.create_dataset("box_translation_base", data=translation_seq)
        h5file.create_dataset("box_6d_rotation_base", data=rotation6d_seq)
        h5file.create_dataset("box_z_degree", data=z_degree_seq)
        h5file.create_dataset("estimation_valid", data=valid_seq)

    return dst_hdf5_path


def process_single_episode(hdf5_path: Path) -> None:
    video_path = hdf5_path.parent / "front_rgb_image.rmb.mp4"
    if not video_path.is_file():
        raise FileNotFoundError(f"front video not found: {video_path}")

    with h5py.File(hdf5_path, "r") as h5file:
        if "time" not in h5file:
            raise KeyError(f"time key is missing in {hdf5_path}.")
        num_steps = int(h5file["time"].shape[0])

    frame_count = _get_frame_count(video_path)
    if frame_count != num_steps:
        raise AssertionError(
            f"{hdf5_path}: frame_count ({frame_count}) != num_steps ({num_steps})."
        )

    translation_seq, rotation6d_seq, z_degree_seq, valid_seq = estimate_box_pose_sequences(
        video_path=video_path,
        num_steps=num_steps,
    )

    dst_hdf5_path = write_eval_hdf5(
        src_hdf5_path=hdf5_path,
        translation_seq=translation_seq,
        rotation6d_seq=rotation6d_seq,
        z_degree_seq=z_degree_seq,
        valid_seq=valid_seq,
    )

    print(f"Created eval hdf5: {dst_hdf5_path}")
    print(
        "Added datasets: box_translation_base, box_6d_rotation_base, box_z_degree, estimation_valid"
    )


def main() -> None:
    args = parse_args()

    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"dataset directory not found: {dataset_dir}")

    hdf5_paths = sorted(dataset_dir.glob("**/main.rmb.hdf5"))
    if len(hdf5_paths) == 0:
        raise FileNotFoundError(f"No main.rmb.hdf5 found under: {dataset_dir}")

    print(f"Found {len(hdf5_paths)} episodes under {dataset_dir}")
    for idx, hdf5_path in enumerate(hdf5_paths, start=1):
        print(f"[{idx}/{len(hdf5_paths)}] Processing {hdf5_path}")
        process_single_episode(hdf5_path)


if __name__ == "__main__":
    main()
