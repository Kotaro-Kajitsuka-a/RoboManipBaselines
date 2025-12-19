from __future__ import annotations

"""
GUI viewer for the ArUco GridBoard-based box pose.

This mirrors bin/box_detection.py but reuses parameters and utilities from the
production SAC task (`dual_box_rotation.py`). The goal is to match the runtime
camera resolution/aspect ratio as closely as possible.
"""

import time
from pathlib import Path
import sys

import cv2
import numpy as np
import pyrealsense2 as rs
from cv2 import aruco

# Support both `python -m robo_manip_baselines...` and `python path/to/box_pose_viewer.py`
if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[3]
    sys.path.append(str(repo_root))
    from robo_manip_baselines.policy.sac.sac_tasks.dual_box_rotation import (
        ARUCO_DICT_ID,
        BASE_CENTER_T_PATH,
        BOX_DEPTH_M,
        MARKER_LENGTH_M,
        MARKER_SEPARATION_M,
        MARKERS_X,
        MARKERS_Y,
        RES_H,
        RES_W,
        FPS,
        USE_SERIAL,
        _rotmat_to_rpy_deg,
    )
else:
    from .dual_box_rotation import (
        ARUCO_DICT_ID,
        BASE_CENTER_T_PATH,
        BOX_DEPTH_M,
        MARKER_LENGTH_M,
        MARKER_SEPARATION_M,
        MARKERS_X,
        MARKERS_Y,
        RES_H,
        RES_W,
        FPS,
        USE_SERIAL,
        _rotmat_to_rpy_deg,
    )


def draw_axes(img, K, rvec, tvec, axis_len=0.05, thickness=2):
    origin = np.float32([[0, 0, 0]])
    axes = np.float32([[axis_len, 0, 0], [0, axis_len, 0], [0, 0, axis_len]])
    pts3d = np.vstack([origin, axes]).reshape(-1, 1, 3)
    proj, _ = cv2.projectPoints(pts3d, rvec, tvec, K, None)
    p0, pX, pY, pZ = [tuple(p.ravel().astype(int)) for p in proj]
    cv2.line(img, p0, pX, (0, 0, 255), thickness)
    cv2.line(img, p0, pY, (0, 255, 0), thickness)
    cv2.line(img, p0, pZ, (255, 0, 0), thickness)


def main():
    base_T_cam = np.loadtxt(Path(BASE_CENTER_T_PATH)).astype(np.float32)

    # ArUco/board setup
    try:
        aruco_dict = aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    except AttributeError:
        aruco_dict = aruco.Dictionary_get(ARUCO_DICT_ID)
    try:
        parameters = aruco.DetectorParameters_create()
    except AttributeError:
        parameters = aruco.DetectorParameters()
    board = aruco.GridBoard(
        (MARKERS_X, MARKERS_Y), MARKER_LENGTH_M, MARKER_SEPARATION_M, aruco_dict
    )
    board_w = MARKERS_X * MARKER_LENGTH_M + (MARKERS_X - 1) * MARKER_SEPARATION_M
    board_h = MARKERS_Y * MARKER_LENGTH_M + (MARKERS_Y - 1) * MARKER_SEPARATION_M

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(USE_SERIAL)
    # Match production: request the same resolution/aspect ratio as the SAC task.
    config.enable_stream(rs.stream.color, RES_W, RES_H, rs.format.rgb8, FPS)
    profile = pipeline.start(config)

    sensor = profile.get_device().first_color_sensor()
    try:
        sensor.set_option(rs.option.enable_auto_exposure, 0)
        sensor.set_option(rs.option.exposure, 140)
        sensor.set_option(rs.option.gain, 64)
    except Exception:
        pass

    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_stream.get_intrinsics()
    fx, fy, cx, cy = intr.fx, intr.fy, intr.ppx, intr.ppy
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.array(intr.coeffs[:5], dtype=np.float32)

    print("Camera intrinsics:")
    print(" fx, fy =", fx, fy)
    print(" cx, cy =", cx, cy)
    print(" dist   =", dist_coeffs)

    R_flip_x = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32)
    center_offset = np.array([board_w * 0.5, board_h * 0.5, 0.0], dtype=np.float32)
    z_offset = np.array([0.0, 0.0, -BOX_DEPTH_M * 0.5], dtype=np.float32)

    t_prev, fps_est = time.time(), 0.0
    last_print = 0.0

    try:
        while True:
            frames = pipeline.wait_for_frames()
            cf = frames.get_color_frame()
            if not cf:
                continue
            # Stream is RGB; convert to BGR for OpenCV drawing/imshow to keep colors correct.
            img = cv2.cvtColor(np.asanyarray(cf.get_data()), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
            board_pose_ok = False
            box_pose_ok = False
            box_center_cam = None
            box_center_base = None
            rpy_base_box = None
            used_ids = []

            if ids is not None and len(ids) > 0:
                used_ids = ids.flatten().tolist()
                aruco.drawDetectedMarkers(img, corners, ids)
                retval, rvec, tvec = aruco.estimatePoseBoard(
                    corners, ids, board, K, dist_coeffs, None, None
                )
                if retval > 0:
                    board_pose_ok = True
                    R_board, _ = cv2.Rodrigues(rvec)
                    rpy_board = _rotmat_to_rpy_deg(R_board)
                    dist_board = float(np.linalg.norm(tvec))

                    board_size_max = max(MARKERS_X, MARKERS_Y) * (
                        MARKER_LENGTH_M + MARKER_SEPARATION_M
                    )
                    draw_axes(img, K, rvec, tvec, axis_len=board_size_max * 0.6, thickness=3)

                    t_board_box = center_offset + R_flip_x @ z_offset
                    R_cam_box = R_board @ R_flip_x
                    t_cam_box = R_board @ t_board_box.reshape(3, 1) + tvec.reshape(3, 1)

                    cam_T_box = np.eye(4, dtype=np.float32)
                    cam_T_box[:3, :3] = R_cam_box.astype(np.float32)
                    cam_T_box[:3, 3] = t_cam_box.flatten().astype(np.float32)
                    base_T_box = base_T_cam @ cam_T_box
                    box_center_cam = t_cam_box.flatten()
                    box_center_base = base_T_box[:3, 3]
                    rpy_base_box = _rotmat_to_rpy_deg(base_T_box[:3, :3])
                    box_pose_ok = True

                    draw_axes(img, K, cv2.Rodrigues(R_cam_box)[0], t_cam_box, axis_len=board_size_max * 0.5, thickness=3)
                    proj_center, _ = cv2.projectPoints(
                        np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
                        cv2.Rodrigues(R_cam_box)[0],
                        t_cam_box,
                        K,
                        None,
                    )
                    cx_int, cy_int = proj_center[0, 0].astype(int)
                    cv2.circle(img, (cx_int, cy_int), 8, (0, 200, 255), -1)

                    if time.time() - last_print > 1.0:
                        print("base->box transform:")
                        print(base_T_box)
                        print(f"box center (base): {box_center_base}")
                        print(f"box rpy (base deg): {rpy_base_box}")
                        print(f"board dist={dist_board:.3f} rpy={rpy_board}")
                        last_print = time.time()

            # UI overlay
            panel_w = 420
            overlay = img.copy()
            cv2.rectangle(overlay, (0, 0), (panel_w, RES_H), (0, 0, 0), -1)
            img = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)
            cv2.putText(
                img,
                f"D435 {RES_W}x{RES_H}@{FPS}  FPS: {fps_est:.1f}",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                img,
                f"Serial: {USE_SERIAL}",
                (12, 54),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (220, 220, 220),
                2,
                cv2.LINE_AA,
            )
            y = 86
            cv2.putText(
                img,
                f"ArUco GridBoard {MARKERS_X}x{MARKERS_Y}  len={MARKER_LENGTH_M*1000:.2f}mm",
                (12, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            y += 24
            cv2.putText(
                img,
                f"Detected markers: {len(used_ids)}",
                (12, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (230, 230, 230),
                2,
                cv2.LINE_AA,
            )
            y += 24
            if board_pose_ok:
                line1 = f"Board pose OK"
                cv2.putText(
                    img,
                    line1,
                    (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.60,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                y += 24
                if box_pose_ok and box_center_cam is not None:
                    line2 = (
                        f"Box center (cam): "
                        f"({box_center_cam[0]:.3f}, {box_center_cam[1]:.3f}, {box_center_cam[2]:.3f}) m"
                    )
                    cv2.putText(
                        img,
                        line2,
                        (12, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (180, 255, 180),
                        2,
                        cv2.LINE_AA,
                    )
                    y += 24
                if box_pose_ok and box_center_base is not None:
                    line3 = (
                        f"Box center (base): "
                        f"({box_center_base[0]:.3f}, {box_center_base[1]:.3f}, {box_center_base[2]:.3f}) m"
                    )
                    cv2.putText(
                        img,
                        line3,
                        (12, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (180, 200, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    y += 24
                    line4 = (
                        f"Box RPY (base): "
                        f"({_rotmat_to_rpy_deg(base_T_box[:3,:3])[0]:.1f}, "
                        f"{_rotmat_to_rpy_deg(base_T_box[:3,:3])[1]:.1f}, "
                        f"{_rotmat_to_rpy_deg(base_T_box[:3,:3])[2]:.1f}) deg"
                    )
                    cv2.putText(
                        img,
                        line4,
                        (12, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (200, 200, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    y += 24
            else:
                cv2.putText(
                    img,
                    "Board pose: ---",
                    (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.60,
                    (200, 200, 200),
                    2,
                    cv2.LINE_AA,
                )
                y += 24

            now = time.time()
            fps_est = 0.9 * fps_est + 0.1 * (1.0 / max(1e-6, now - t_prev))
            t_prev = now

            cv2.imshow("Box Pose Viewer", img)
            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord("q")):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
