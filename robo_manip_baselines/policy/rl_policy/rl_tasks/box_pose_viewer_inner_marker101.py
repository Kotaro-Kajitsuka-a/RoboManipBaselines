from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from robo_manip_baselines.policy.rl_policy.rl_tasks.cabinet_marker_detection import (
    BASE_CENTER_T_PATH,
    FPS,
    RES_H,
    RES_W,
    SMALL_BOARD_DICT_ID,
    SMALL_BOARD_MARKER_LENGTH_M,
    SMALL_BOARD_TARGET_RY_OFFSET_RAD,
    SMALL_BOARD_TARGET_Y_OFFSET_M,
    USE_SERIAL,
    detect_markers,
    get_aruco_dictionary,
    get_aruco_parameters,
    rotation_y,
)

WORKSPACE_CENTER = np.array([-0.615, 0.0, 0.088], dtype=np.float32)
INNER_CENTER_MARKER_ID = 101


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


def draw_axes(img, K, dist_coeffs, rvec, tvec, axis_len=0.05, thickness=2):
    origin = np.float32([[0, 0, 0]])
    axes = np.float32([[axis_len, 0, 0], [0, axis_len, 0], [0, 0, axis_len]])
    pts3d = np.vstack([origin, axes]).reshape(-1, 1, 3)
    proj, _ = cv2.projectPoints(pts3d, rvec, tvec, K, dist_coeffs)
    p0, p_x, p_y, p_z = [tuple(p.ravel().astype(int)) for p in proj]
    cv2.line(img, p0, p_x, (0, 0, 255), thickness)
    cv2.line(img, p0, p_y, (0, 255, 0), thickness)
    cv2.line(img, p0, p_z, (255, 0, 0), thickness)


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


def estimate_marker101_inner_pose(corners, ids, K, dist_coeffs):
    if ids is None or len(ids) == 0:
        return None, None, None, None

    ids_flat = ids.reshape(-1)
    matched = np.where(ids_flat == INNER_CENTER_MARKER_ID)[0]
    if matched.size == 0:
        return None, None, None, None

    marker_corners = [corners[int(matched[0])]]
    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
        marker_corners,
        SMALL_BOARD_MARKER_LENGTH_M,
        K,
        dist_coeffs,
    )
    marker_rvec = rvecs[0].reshape(3, 1).astype(np.float32)
    marker_tvec = tvecs[0].reshape(3, 1).astype(np.float32)

    # Treat marker 101 center as the mean position of markers [100, 101, 102].
    R_marker, _ = cv2.Rodrigues(marker_rvec)
    target_offset = np.array(
        [0.0, SMALL_BOARD_TARGET_Y_OFFSET_M, 0.0], dtype=np.float32
    )
    R_inner = R_marker @ rotation_y(SMALL_BOARD_TARGET_RY_OFFSET_RAD)
    t_inner = R_marker @ target_offset.reshape(3, 1) + marker_tvec
    inner_rvec, _ = cv2.Rodrigues(R_inner)
    return (
        marker_rvec,
        marker_tvec,
        inner_rvec.reshape(3, 1).astype(np.float32),
        t_inner.reshape(3, 1).astype(np.float32),
    )


def main():
    import pyrealsense2 as rs

    base_T_cam = np.loadtxt(Path(BASE_CENTER_T_PATH)).astype(np.float32)
    small_dict = get_aruco_dictionary(SMALL_BOARD_DICT_ID)
    params = get_aruco_parameters()

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
    print("workspace center =", WORKSPACE_CENTER)
    print(
        f"Using marker {INNER_CENTER_MARKER_ID} as inner small-board center. "
        f"target_y_offset={SMALL_BOARD_TARGET_Y_OFFSET_M} m, "
        f"target_ry_offset={np.rad2deg(SMALL_BOARD_TARGET_RY_OFFSET_RAD)} deg"
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
            inner_policy = None
            inner_rpy = None
            detected_ids = []

            corners, ids, _ = detect_markers(gray, small_dict, params)
            if ids is not None and len(ids) > 0:
                detected_ids = ids.reshape(-1).astype(int).tolist()
                cv2.aruco.drawDetectedMarkers(img, corners, ids)
                marker_rvec, marker_tvec, inner_rvec, inner_tvec = (
                    estimate_marker101_inner_pose(corners, ids, K, dist_coeffs)
                )
                if inner_rvec is not None and inner_tvec is not None:
                    draw_axes(
                        img,
                        K,
                        dist_coeffs,
                        marker_rvec,
                        marker_tvec,
                        axis_len=SMALL_BOARD_MARKER_LENGTH_M,
                        thickness=2,
                    )
                    draw_axes(
                        img,
                        K,
                        dist_coeffs,
                        inner_rvec,
                        inner_tvec,
                        axis_len=SMALL_BOARD_MARKER_LENGTH_M * 1.8,
                        thickness=4,
                    )
                    project_center(
                        img,
                        K,
                        dist_coeffs,
                        inner_rvec,
                        inner_tvec,
                        (255, 0, 255),
                        "inner from 101",
                    )

                    R_inner, _ = cv2.Rodrigues(inner_rvec)
                    cam_T_inner = np.eye(4, dtype=np.float32)
                    cam_T_inner[:3, :3] = R_inner.astype(np.float32)
                    cam_T_inner[:3, 3] = inner_tvec.flatten().astype(np.float32)
                    base_T_inner = base_T_cam @ cam_T_inner
                    inner_base = base_T_inner[:3, 3]
                    inner_policy = inner_base - WORKSPACE_CENTER
                    inner_rpy = rotmat_to_rpy_deg(base_T_inner[:3, :3])

            now = time.time()
            fps_est = 0.9 * fps_est + 0.1 * (1.0 / max(1e-6, now - t_prev))
            t_prev = now

            overlay = img.copy()
            panel_w = 620
            cv2.rectangle(overlay, (0, 0), (panel_w, RES_H), (0, 0, 0), -1)
            img = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)
            y = 30
            y = put_line(img, f"Inner marker101 viewer  FPS: {fps_est:.1f}", y)
            y = put_line(img, f"Serial: {USE_SERIAL}", y, (220, 220, 220))
            y = put_line(img, f"Workspace center: {WORKSPACE_CENTER.tolist()}", y)
            y = put_line(img, f"Detected small IDs: {detected_ids}", y)
            y += 8
            y = put_line(
                img,
                f"Marker {INNER_CENTER_MARKER_ID} as small-board center: "
                f"{'OK' if inner_base is not None else '---'}",
                y,
                (255, 160, 255) if inner_base is not None else (180, 180, 180),
            )
            y = put_line(
                img,
                f"  offset: y={SMALL_BOARD_TARGET_Y_OFFSET_M:.4f} m, "
                f"ry={np.rad2deg(SMALL_BOARD_TARGET_RY_OFFSET_RAD):.1f} deg",
                y,
            )
            if inner_base is not None:
                y = put_line(img, f"  base:   {np.round(inner_base, 4)}", y)
                y = put_line(img, f"  policy: {np.round(inner_policy, 4)}", y)
                y = put_line(img, f"  rpy:    {np.round(inner_rpy, 2)} deg", y)

            if time.time() - last_print > 1.0:
                if inner_base is not None:
                    print(
                        f"[InnerMarker101] base={inner_base} "
                        f"policy={inner_policy} rpy={inner_rpy}",
                        flush=True,
                    )
                else:
                    print(
                        f"[InnerMarker101] marker {INNER_CENTER_MARKER_ID} not detected. "
                        f"detected_ids={detected_ids}",
                        flush=True,
                    )
                last_print = time.time()

            cv2.imshow("RLPolicy Inner Marker101 Viewer", img)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
