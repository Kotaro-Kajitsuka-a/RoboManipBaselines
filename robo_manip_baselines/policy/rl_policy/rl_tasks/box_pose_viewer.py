from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
from cv2 import aruco

from robo_manip_baselines.policy.rl_policy.rl_tasks.cabinet_marker_detection import (
    BASE_CENTER_T_PATH,
    BIG_ARUCO_DICT_ID,
    BIG_MARKER_LENGTH_M,
    BIG_MARKER_SEPARATION_M,
    BIG_MARKERS_X,
    BIG_MARKERS_Y,
    FPS,
    RES_H,
    RES_W,
    SMALL_BOARD_DICT_ID,
    SMALL_BOARD_MARKER_IDS,
    SMALL_BOARD_MARKER_LENGTH_M,
    USE_SERIAL,
    build_big_board,
    build_small_board,
    detect_markers,
    estimate_big_board_outer_pose,
    estimate_small_board_pose,
    get_aruco_dictionary,
    get_aruco_parameters,
    transform_from_rvec_tvec,
)


def draw_axes(img, K, dist_coeffs, rvec, tvec, axis_len=0.05, thickness=2):
    origin = np.float32([[0, 0, 0]])
    axes = np.float32([[axis_len, 0, 0], [0, axis_len, 0], [0, 0, axis_len]])
    pts3d = np.vstack([origin, axes]).reshape(-1, 1, 3)
    proj, _ = cv2.projectPoints(pts3d, rvec, tvec, K, dist_coeffs)
    p0, p_x, p_y, p_z = [tuple(p.ravel().astype(int)) for p in proj]
    cv2.line(img, p0, p_x, (0, 0, 255), thickness)
    cv2.line(img, p0, p_y, (0, 255, 0), thickness)
    cv2.line(img, p0, p_z, (255, 0, 0), thickness)


def rotmat_to_rpy_deg(rotation: np.ndarray) -> np.ndarray:
    sy = -rotation[2, 0]
    sy = np.clip(sy, -1.0, 1.0)
    pitch = np.arcsin(sy)
    if abs(sy) < 0.999999:
        roll = np.arctan2(rotation[2, 1], rotation[2, 2])
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = np.arctan2(-rotation[1, 2], rotation[1, 1])
        yaw = 0.0
    return np.degrees([roll, pitch, yaw])


def put_line(img, text, y, color=(235, 235, 235), scale=0.54):
    cv2.putText(
        img,
        text,
        (12, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        2,
        cv2.LINE_AA,
    )
    return y + 24


def project_center(img, K, dist_coeffs, rvec, tvec, color, label):
    center_px, _ = cv2.projectPoints(
        np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        rvec,
        tvec,
        K,
        dist_coeffs,
    )
    cx, cy = center_px[0, 0].astype(int)
    cv2.circle(img, (cx, cy), 8, color, -1)
    cv2.putText(
        img,
        label,
        (cx + 10, cy - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA,
    )


def main():
    import pyrealsense2 as rs

    base_T_cam = np.loadtxt(Path(BASE_CENTER_T_PATH)).astype(np.float32)
    cam_T_base = np.linalg.inv(base_T_cam)

    R_cam_base = cam_T_base[:3, :3]
    t_cam_base = cam_T_base[:3, 3]

    base_rvec, _ = cv2.Rodrigues(R_cam_base)
    base_tvec = t_cam_base.reshape(3, 1)

    big_dict = get_aruco_dictionary(BIG_ARUCO_DICT_ID)
    small_dict = get_aruco_dictionary(SMALL_BOARD_DICT_ID)
    params = get_aruco_parameters()
    small_board = build_small_board(small_dict)
    big_board = build_big_board(big_dict)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(USE_SERIAL)
    config.enable_stream(rs.stream.color, RES_W, RES_H, rs.format.bgr8, FPS)
    profile = pipeline.start(config)

    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_stream.get_intrinsics()
    K = np.array(
        [[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]],
        dtype=np.float32,
    )
    dist_coeffs = np.array(intr.coeffs[:5], dtype=np.float32)

    print("Camera intrinsics:")
    print(" fx, fy =", intr.fx, intr.fy)
    print(" cx, cy =", intr.ppx, intr.ppy)
    print(" dist   =", dist_coeffs)

    big_axis_len = (
        max(BIG_MARKERS_X, BIG_MARKERS_Y)
        * (BIG_MARKER_LENGTH_M + BIG_MARKER_SEPARATION_M)
        * 0.45
    )

    t_prev = time.time()
    fps_est = 0.0
    last_print = 0.0

    try:
        while True:
            frames = pipeline.wait_for_frames()
            cf = frames.get_color_frame()
            if not cf:
                continue

            img = np.asanyarray(cf.get_data())
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            inner_base = None
            inner_rpy = None
            outer_base = None
            outer_rpy = None
            outer_marker_count = 0

            small_corners, small_ids, _ = detect_markers(gray, small_dict, params)
            small_rvec, small_tvec = estimate_small_board_pose(
                small_corners,
                small_ids,
                small_board,
                K,
                dist_coeffs,
            )
            if small_rvec is not None and small_tvec is not None:
                aruco.drawDetectedMarkers(img, small_corners, small_ids)
                draw_axes(
                    img,
                    K,
                    dist_coeffs,
                    small_rvec,
                    small_tvec,
                    axis_len=SMALL_BOARD_MARKER_LENGTH_M * 1.8,
                    thickness=4,
                )
                project_center(
                    img,
                    K,
                    dist_coeffs,
                    small_rvec,
                    small_tvec,
                    (255, 0, 255),
                    "inner",
                )

                draw_axes(
                    img,
                    K,
                    dist_coeffs,
                    base_rvec,
                    base_tvec,
                    axis_len=0.10,
                    thickness=4,
                )

                project_center(
                    img,
                    K,
                    dist_coeffs,
                    base_rvec,
                    base_tvec,
                    (0, 255, 255),
                    "base",
                )

                cam_T_inner = transform_from_rvec_tvec(small_rvec, small_tvec)

                base_T_inner = base_T_cam @ cam_T_inner
                base_T_inner = base_T_cam @ cam_T_inner
                inner_base = base_T_inner[:3, 3]
                inner_rpy = rotmat_to_rpy_deg(base_T_inner[:3, :3])

            corners, ids, _ = detect_markers(gray, big_dict, params)
            if ids is not None and len(ids) > 0:
                outer_marker_count = len(ids)
                aruco.drawDetectedMarkers(img, corners, ids)
                outer_rvec, outer_tvec = estimate_big_board_outer_pose(
                    corners,
                    ids,
                    big_board,
                    K,
                    dist_coeffs,
                )
                if outer_rvec is not None and outer_tvec is not None:
                    draw_axes(
                        img,
                        K,
                        dist_coeffs,
                        outer_rvec,
                        outer_tvec,
                        axis_len=big_axis_len,
                        thickness=3,
                    )
                    project_center(
                        img,
                        K,
                        dist_coeffs,
                        outer_rvec,
                        outer_tvec,
                        (0, 200, 255),
                        "outer",
                    )

                    cam_T_outer = transform_from_rvec_tvec(outer_rvec, outer_tvec)
                    base_T_outer = base_T_cam @ cam_T_outer
                    outer_base = base_T_outer[:3, 3]
                    outer_rpy = rotmat_to_rpy_deg(base_T_outer[:3, :3])

            now = time.time()
            fps_est = 0.9 * fps_est + 0.1 * (1.0 / max(1e-6, now - t_prev))
            t_prev = now

            overlay = img.copy()
            panel_w = 620
            cv2.rectangle(overlay, (0, 0), (panel_w, RES_H), (0, 0, 0), -1)
            img = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)
            y = 30
            y = put_line(img, f"RLPolicy marker viewer  FPS: {fps_est:.1f}", y)
            y = put_line(img, f"Serial: {USE_SERIAL}", y, (220, 220, 220))
            y += 8
            y = put_line(
                img,
                f"Outer GridBoard {BIG_MARKERS_X}x{BIG_MARKERS_Y}: markers={outer_marker_count}",
                y,
                (0, 220, 255) if outer_base is not None else (180, 180, 180),
            )
            if outer_base is not None:
                y = put_line(img, f"  base:   {np.round(outer_base, 4)}", y)
                y = put_line(img, f"  rpy:    {np.round(outer_rpy, 2)} deg", y)
            y += 8
            y = put_line(
                img,
                f"Inner SmallBoard {SMALL_BOARD_MARKER_IDS}: "
                f"{'OK' if inner_base is not None else '---'}",
                y,
                (255, 160, 255) if inner_base is not None else (180, 180, 180),
            )
            if inner_base is not None:
                y = put_line(img, f"  base:   {np.round(inner_base, 4)}", y)
                y = put_line(img, f"  rpy:    {np.round(inner_rpy, 2)} deg", y)

            if time.time() - last_print > 1.0:
                if outer_base is not None:
                    print(
                        f"[OuterBoard] base={outer_base} rpy={outer_rpy}",
                        flush=True,
                    )
                if inner_base is not None:
                    print(
                        f"[InnerBoard] base={inner_base} rpy={inner_rpy}",
                        flush=True,
                    )
                last_print = time.time()

            cv2.imshow("RLPolicy Box Pose Viewer", img)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
