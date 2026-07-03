from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
from cv2 import aruco

from robo_manip_baselines.policy.rl_policy.rl_tasks.cabinet_marker_detection import (
    BASE_CENTER_T_PATH,
    FPS,
    RES_H,
    RES_W,
    USE_SERIAL,
)

MARKER_ID = 0
MARKER_SIZE_M = 0.09265


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


def project_marker_center(img, K, dist_coeffs, rvec, tvec, color, label):
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


def estimate_marker_pose_from_corners(marker_corners, marker_size_m, K, dist_coeffs):
    half = 0.5 * float(marker_size_m)
    object_points = np.array(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float32,
    )
    image_points = np.asarray(marker_corners, dtype=np.float32).reshape(4, 2)
    ok, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        K,
        dist_coeffs,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not ok:
        return None, None
    return rvec.astype(np.float32), tvec.astype(np.float32)


def estimate_marker_translation(corners, ids, K, dist_coeffs):
    if ids is None or len(ids) == 0:
        return None, None, None

    ids_flat = ids.reshape(-1)
    matched = np.where(ids_flat == MARKER_ID)[0]
    if matched.size == 0:
        return None, None, None

    marker_corners = [corners[int(matched[0])]]
    rvec, tvec = estimate_marker_pose_from_corners(
        marker_corners[0], MARKER_SIZE_M, K, dist_coeffs
    )
    if rvec is None or tvec is None:
        return None, None, None
    return rvec, tvec, tvec.reshape(3)


def main():
    import pyrealsense2 as rs

    base_T_cam = np.loadtxt(Path(BASE_CENTER_T_PATH)).astype(np.float32)
    marker_dict = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
    params = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(marker_dict, params)

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
    print(f"tag36h11 marker id={MARKER_ID}, size={MARKER_SIZE_M} m")

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

            marker_base = None
            detected_ids = []

            corners, ids, _ = detector.detectMarkers(gray)
            if ids is not None and len(ids) > 0:
                detected_ids = ids.reshape(-1).astype(int).tolist()
                aruco.drawDetectedMarkers(img, corners, ids)
                rvec, tvec, marker_cam = estimate_marker_translation(
                    corners, ids, K, dist_coeffs
                )
                if marker_cam is not None:
                    cam_pos_h = np.ones(4, dtype=np.float32)
                    cam_pos_h[:3] = marker_cam
                    marker_base = (base_T_cam @ cam_pos_h)[:3]
                    project_marker_center(
                        img,
                        K,
                        dist_coeffs,
                        rvec,
                        tvec,
                        (255, 0, 255),
                        f"tag {MARKER_ID}",
                    )

            now = time.time()
            fps_est = 0.9 * fps_est + 0.1 * (1.0 / max(1e-6, now - t_prev))
            t_prev = now

            overlay = img.copy()
            panel_w = 560
            cv2.rectangle(overlay, (0, 0), (panel_w, RES_H), (0, 0, 0), -1)
            img = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)
            y = 30
            y = put_line(img, f"Tag36h11 id {MARKER_ID} translation  FPS: {fps_est:.1f}", y)
            y = put_line(img, f"Serial: {USE_SERIAL}", y, (220, 220, 220))
            y = put_line(img, f"Marker size: {MARKER_SIZE_M:.4f} m", y)
            y = put_line(img, f"Detected IDs: {detected_ids}", y)
            y += 8
            y = put_line(
                img,
                f"Tag {MARKER_ID}: {'OK' if marker_base is not None else '---'}",
                y,
                (255, 160, 255) if marker_base is not None else (180, 180, 180),
            )
            if marker_base is not None:
                y = put_line(img, f"  base translation: {np.round(marker_base, 4)}", y)

            if time.time() - last_print > 1.0:
                if marker_base is not None:
                    print(
                        f"[AprilTag{MARKER_ID}] base_translation={marker_base}",
                        flush=True,
                    )
                else:
                    print(
                        f"[AprilTag{MARKER_ID}] not detected. detected_ids={detected_ids}",
                        flush=True,
                    )
                last_print = time.time()

            cv2.imshow(f"Tag36h11 {MARKER_ID} Translation Viewer", img)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
