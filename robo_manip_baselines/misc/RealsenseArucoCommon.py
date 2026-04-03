import argparse
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from cv2 import aruco


def _get_aruco_dictionary(dict_name: str):
    dict_id = getattr(aruco, dict_name)
    try:
        return aruco.getPredefinedDictionary(dict_id)
    except AttributeError:
        return aruco.Dictionary_get(dict_id)


def _create_detector_parameters():
    try:
        return aruco.DetectorParameters_create()
    except AttributeError:
        return aruco.DetectorParameters()


def _build_argument_parser(
    default_width: int, default_height: int
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Detect ArUco markers from Intel RealSense D435i color stream and measure "
            "the farthest distance that still yields detection."
        )
    )
    parser.add_argument(
        "--serial",
        type=str,
        default=None,
        help="RealSense serial number. If omitted, the first detected device is used.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Requested color FPS for the RealSense stream.",
    )
    parser.add_argument(
        "--marker_length_mm",
        type=float,
        required=True,
        help="Physical side length of one ArUco marker in millimeters.",
    )
    parser.add_argument(
        "--dict_name",
        type=str,
        default="DICT_4X4_50",
        help="OpenCV ArUco dictionary name, e.g. DICT_4X4_50.",
    )
    parser.add_argument(
        "--target_id",
        type=int,
        default=None,
        help="If specified, only this marker ID is used for distance reporting.",
    )
    parser.add_argument(
        "--show_rejected",
        action="store_true",
        help="Draw rejected marker candidates for debugging.",
    )
    parser.add_argument(
        "--window_name",
        type=str,
        default=f"RealSense ArUco {default_width}x{default_height}",
        help="OpenCV window title.",
    )
    parser.add_argument(
        "--display_scale",
        type=float,
        default=1.0,
        help="Scale factor applied only to the displayed image.",
    )
    return parser


def _marker_edge_length_px(corner: np.ndarray) -> float:
    pts = np.asarray(corner, dtype=np.float32).reshape(4, 2)
    edges = [np.linalg.norm(pts[(i + 1) % 4] - pts[i]) for i in range(4)]
    return float(np.mean(edges))


def _put_line(
    frame: np.ndarray,
    text: str,
    line_idx: int,
    color=(255, 255, 255),
    font_scale: float = 0.5,
    thickness: int = 1,
    line_height: int = 20,
) -> None:
    x = 12
    y = 24 + line_height * line_idx
    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


@dataclass
class DetectionRecord:
    marker_id: int
    distance_m: float
    edge_px: float
    tvec: np.ndarray


def run_realsense_aruco(default_width: int, default_height: int) -> None:
    parser = _build_argument_parser(default_width, default_height)
    args = parser.parse_args()

    assert args.marker_length_mm > 0.0
    assert args.display_scale > 0.0
    marker_length_m = 1e-3 * args.marker_length_mm

    try:
        import pyrealsense2 as rs  # type: ignore
    except Exception as exc:
        raise RuntimeError("pyrealsense2 is required for these scripts.") from exc

    aruco_dict = _get_aruco_dictionary(args.dict_name)
    detector_params = _create_detector_parameters()

    pipeline = rs.pipeline()
    config = rs.config()
    if args.serial is not None:
        config.enable_device(args.serial)
    config.enable_stream(
        rs.stream.color, default_width, default_height, rs.format.bgr8, args.fps
    )

    profile = pipeline.start(config)
    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_profile.get_intrinsics()
    width = int(intr.width)
    height = int(intr.height)
    K = np.array(
        [[intr.fx, 0.0, intr.ppx], [0.0, intr.fy, intr.ppy], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    dist_coeffs = np.array(intr.coeffs[:5], dtype=np.float32)

    print(
        f"[RealsenseAruco] stream={width}x{height}@{args.fps} "
        f"serial={args.serial or 'auto'} marker_length_mm={args.marker_length_mm:.3f} "
        f"dict={args.dict_name} target_id={args.target_id} "
        f"display_scale={args.display_scale:.2f}"
    )
    print(
        f"[RealsenseAruco] fx={intr.fx:.3f} fy={intr.fy:.3f} "
        f"cx={intr.ppx:.3f} cy={intr.ppy:.3f}"
    )
    print("[RealsenseAruco] Press q to quit.")

    best_record: Optional[DetectionRecord] = None
    last_report_time = 0.0
    last_frame_time = time.time()
    fps_ema = 0.0
    base_scale = min(width / 1280.0, height / 720.0)
    text_scale = max(0.45, min(0.7, 0.55 * base_scale))
    text_thickness = 1 if text_scale < 0.62 else 2
    line_height = max(18, int(34 * text_scale))
    panel_height = 28 + 5 * line_height

    try:
        cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL)
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())
            now = time.time()
            dt = max(now - last_frame_time, 1e-6)
            last_frame_time = now
            fps_now = 1.0 / dt
            if fps_ema == 0.0:
                fps_ema = fps_now
            else:
                fps_ema = 0.9 * fps_ema + 0.1 * fps_now

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, rejected = aruco.detectMarkers(
                gray, aruco_dict, parameters=detector_params
            )

            current_record: Optional[DetectionRecord] = None
            detected_ids = []
            if ids is not None and len(ids) > 0:
                ids_flat = ids.flatten().astype(int)
                selected_indices = [
                    idx
                    for idx, marker_id in enumerate(ids_flat)
                    if args.target_id is None or marker_id == args.target_id
                ]
                if selected_indices:
                    selected_corners = [corners[idx] for idx in selected_indices]
                    selected_ids = ids[selected_indices]
                    detected_ids = [int(v) for v in selected_ids.flatten()]
                    aruco.drawDetectedMarkers(frame, selected_corners, selected_ids)

                    rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                        selected_corners, marker_length_m, K, dist_coeffs
                    )
                    for marker_id, corner, rvec, tvec in zip(
                        detected_ids, selected_corners, rvecs, tvecs
                    ):
                        tvec = np.asarray(tvec, dtype=np.float32).reshape(3)
                        edge_px = _marker_edge_length_px(corner)
                        distance_m = float(np.linalg.norm(tvec))
                        cv2.drawFrameAxes(
                            frame,
                            K,
                            dist_coeffs,
                            rvec,
                            tvec.reshape(3, 1),
                            marker_length_m * 0.5,
                        )
                        center = np.mean(np.asarray(corner).reshape(4, 2), axis=0)
                        label = (
                            f"id={marker_id} d={distance_m:.3f}m edge={edge_px:.1f}px"
                        )
                        cv2.putText(
                            frame,
                            label,
                            tuple(np.round(center).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (0, 255, 0),
                            2,
                            cv2.LINE_AA,
                        )
                        record = DetectionRecord(
                            marker_id=marker_id,
                            distance_m=distance_m,
                            edge_px=edge_px,
                            tvec=tvec.copy(),
                        )
                        if (
                            current_record is None
                            or record.distance_m > current_record.distance_m
                        ):
                            current_record = record

                    if current_record is not None and (
                        best_record is None
                        or current_record.distance_m > best_record.distance_m
                    ):
                        best_record = current_record
                else:
                    detected_ids = [int(v) for v in ids_flat]
                    aruco.drawDetectedMarkers(frame, corners, ids)

            if args.show_rejected and rejected is not None and len(rejected) > 0:
                for candidate in rejected:
                    pts = np.round(np.asarray(candidate).reshape(4, 2)).astype(int)
                    cv2.polylines(frame, [pts], True, (0, 0, 255), 1)

            panel = frame.copy()
            cv2.rectangle(panel, (0, 0), (min(width, 900), panel_height), (0, 0, 0), -1)
            frame = cv2.addWeighted(panel, 0.35, frame, 0.65, 0)
            _put_line(
                frame,
                f"stream={width}x{height}@{args.fps} fps={fps_ema:.1f}",
                0,
                font_scale=text_scale,
                thickness=text_thickness,
                line_height=line_height,
            )
            _put_line(
                frame,
                f"marker_length={args.marker_length_mm:.2f} mm dict={args.dict_name}",
                1,
                font_scale=text_scale,
                thickness=text_thickness,
                line_height=line_height,
            )
            _put_line(
                frame,
                f"target_id={args.target_id if args.target_id is not None else 'all'} detected_ids={detected_ids}",
                2,
                font_scale=text_scale,
                thickness=text_thickness,
                line_height=line_height,
            )
            if current_record is None:
                _put_line(
                    frame,
                    "current: no marker detected",
                    3,
                    color=(0, 200, 255),
                    font_scale=text_scale,
                    thickness=text_thickness,
                    line_height=line_height,
                )
            else:
                _put_line(
                    frame,
                    (
                        f"current: id={current_record.marker_id} "
                        f"distance={current_record.distance_m:.3f} m "
                        f"edge={current_record.edge_px:.1f} px"
                    ),
                    3,
                    color=(0, 255, 0),
                    font_scale=text_scale,
                    thickness=text_thickness,
                    line_height=line_height,
                )
            if best_record is None:
                _put_line(
                    frame,
                    "best: none yet",
                    4,
                    color=(255, 200, 0),
                    font_scale=text_scale,
                    thickness=text_thickness,
                    line_height=line_height,
                )
            else:
                _put_line(
                    frame,
                    (
                        f"best: id={best_record.marker_id} "
                        f"distance={best_record.distance_m:.3f} m "
                        f"edge={best_record.edge_px:.1f} px"
                    ),
                    4,
                    color=(255, 200, 0),
                    font_scale=text_scale,
                    thickness=text_thickness,
                    line_height=line_height,
                )

            if args.display_scale == 1.0:
                display_frame = frame
            else:
                display_frame = cv2.resize(
                    frame,
                    dsize=None,
                    fx=args.display_scale,
                    fy=args.display_scale,
                    interpolation=cv2.INTER_LINEAR,
                )

            cv2.imshow(args.window_name, display_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            if now - last_report_time >= 1.0:
                if current_record is None:
                    print(
                        f"[RealsenseAruco] {width}x{height}: no marker detected "
                        f"(target_id={args.target_id}, ids_seen={detected_ids})"
                    )
                else:
                    print(
                        f"[RealsenseAruco] {width}x{height}: current "
                        f"id={current_record.marker_id} "
                        f"distance={current_record.distance_m:.3f}m "
                        f"edge={current_record.edge_px:.1f}px "
                        f"best={best_record.distance_m:.3f}m"
                    )
                last_report_time = now
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    if best_record is None:
        print("[RealsenseAruco] summary: no marker was detected.")
    else:
        print(
            "[RealsenseAruco] summary: "
            f"best_distance={best_record.distance_m:.3f}m "
            f"marker_id={best_record.marker_id} "
            f"edge={best_record.edge_px:.1f}px "
            f"tvec={best_record.tvec.tolist()}"
        )
