from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys

import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[3]
    sys.path.append(str(repo_root))
    from robo_manip_baselines.policy.sac.sac_tasks.dual_box_rotation import (
        BASE_CENTER_T_PATH,
        BOX_DEPTH_M,
        MARKER_LENGTH_M,
        MARKER_SEPARATION_M,
        MARKERS_X,
        MARKERS_Y,
        GRIDBOARD_H_M,
        GRIDBOARD_W_M,
        RES_H,
        RES_W,
        FPS,
        USE_SERIAL,
        _rotmat_to_rpy_deg,
    )
else:
    from .dual_box_rotation import (
        BASE_CENTER_T_PATH,
        BOX_DEPTH_M,
        MARKER_LENGTH_M,
        MARKER_SEPARATION_M,
        MARKERS_X,
        MARKERS_Y,
        GRIDBOARD_H_M,
        GRIDBOARD_W_M,
        RES_H,
        RES_W,
        FPS,
        USE_SERIAL,
        _rotmat_to_rpy_deg,
    )

KEYPOINT_ORDER = [0, 1, 2, 3]


def _solve_pnp_from_keypoints(keypoints_xy, K, dist_coeffs):
    if keypoints_xy is None or len(keypoints_xy) < 4:
        return None

    img_pts = np.asarray([keypoints_xy[i] for i in KEYPOINT_ORDER], dtype=np.float32).reshape(-1, 2)
    obj_pts = np.array(
        [
            [0.0, 0.0, 0.0],
            [GRIDBOARD_W_M, 0.0, 0.0],
            [GRIDBOARD_W_M, GRIDBOARD_H_M, 0.0],
            [0.0, GRIDBOARD_H_M, 0.0],
        ],
        dtype=np.float32,
    )
    ok, rvec, tvec = cv2.solvePnP(
        obj_pts,
        img_pts,
        K,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None
    return rvec, tvec


def draw_axes(img, K, rvec, tvec, axis_len=0.05, thickness=2):
    origin = np.float32([[0, 0, 0]])
    axes = np.float32([[axis_len, 0, 0], [0, axis_len, 0], [0, 0, axis_len]])
    pts3d = np.vstack([origin, axes]).reshape(-1, 1, 3)
    proj, _ = cv2.projectPoints(pts3d, rvec, tvec, K, None)
    p0, pX, pY, pZ = [tuple(p.ravel().astype(int)) for p in proj]
    cv2.line(img, p0, pX, (0, 0, 255), thickness)
    cv2.line(img, p0, pY, (0, 255, 0), thickness)
    cv2.line(img, p0, pZ, (255, 0, 0), thickness)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize YOLO PnP-based box pose.")
    parser.add_argument("pt_path", help="Path to .pt model file.")
    args = parser.parse_args()

    base_T_cam = np.loadtxt(Path(BASE_CENTER_T_PATH).expanduser().resolve()).astype(np.float32)

    model = YOLO(args.pt_path)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(USE_SERIAL)
    config.enable_stream(rs.stream.color, RES_W, RES_H, rs.format.rgb8, FPS)
    profile = pipeline.start(config)

    # Match normal record-data behavior: keep camera-side default exposure/gain control.

    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_stream.get_intrinsics()
    fx, fy, cx, cy = intr.fx, intr.fy, intr.ppx, intr.ppy
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.array(intr.coeffs[:5], dtype=np.float32)

    R_flip_x = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32)
    center_offset = np.array([GRIDBOARD_W_M * 0.5, GRIDBOARD_H_M * 0.5, 0.0], dtype=np.float32)
    z_offset = np.array([0.0, 0.0, -BOX_DEPTH_M * 0.5], dtype=np.float32)

    t_prev = time.time()
    fps_est = 0.0
    last_print = 0.0

    try:
        while True:
            frames = pipeline.wait_for_frames()
            cf = frames.get_color_frame()
            if not cf:
                continue

            img = cv2.cvtColor(np.asanyarray(cf.get_data()), cv2.COLOR_RGB2BGR)

            results = model.track(img, persist=True, verbose=False)
            annotated = results[0].plot()

            keypoints = results[0].keypoints
            box_center_base = None
            rpy_base_box = None
            if keypoints is not None and keypoints.xy is not None and len(keypoints.xy) > 0:
                pts = keypoints.xy[0].detach().cpu().numpy()
                pnp = _solve_pnp_from_keypoints(pts, K, dist_coeffs)
                if pnp is not None:
                    rvec, tvec = pnp
                    board_size_max = max(MARKERS_X, MARKERS_Y) * (
                        MARKER_LENGTH_M + MARKER_SEPARATION_M
                    )
                    draw_axes(annotated, K, rvec, tvec, axis_len=board_size_max * 0.6, thickness=3)

                    R_board, _ = cv2.Rodrigues(rvec)
                    t_board_box = center_offset + R_flip_x @ z_offset
                    R_cam_box = R_board @ R_flip_x
                    t_cam_box = R_board @ t_board_box.reshape(3, 1) + tvec.reshape(3, 1)

                    cam_T_box = np.eye(4, dtype=np.float32)
                    cam_T_box[:3, :3] = R_cam_box.astype(np.float32)
                    cam_T_box[:3, 3] = t_cam_box.flatten().astype(np.float32)
                    base_T_box = base_T_cam @ cam_T_box
                    box_center_base = base_T_box[:3, 3]
                    rpy_base_box = _rotmat_to_rpy_deg(base_T_box[:3, :3])

                    rvec_box, _ = cv2.Rodrigues(R_cam_box)
                    draw_axes(annotated, K, rvec_box, t_cam_box, axis_len=board_size_max * 0.5, thickness=3)

                    if time.time() - last_print > 1.0:
                        print("t_base_box:", base_T_box[:3, 3])
                        last_print = time.time()

            now = time.time()
            if now - t_prev > 0.2:
                fps_est = 1.0 / max(now - t_prev, 1e-6)
                t_prev = now

            panel_w = 420
            overlay = annotated.copy()
            cv2.rectangle(overlay, (0, 0), (panel_w, RES_H), (0, 0, 0), -1)
            annotated = cv2.addWeighted(overlay, 0.35, annotated, 0.65, 0)
            cv2.putText(
                annotated,
                f"D435 {RES_W}x{RES_H}@{FPS}  FPS: {fps_est:.1f}",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated,
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
                annotated,
                f"YOLO PnP GridBoard {MARKERS_X}x{MARKERS_Y}  len={MARKER_LENGTH_M*1000:.2f}mm",
                (12, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            y += 24
            if box_center_base is not None:
                cv2.putText(
                    annotated,
                    f"box center (base): [{box_center_base[0]:.3f}, {box_center_base[1]:.3f}, {box_center_base[2]:.3f}]",
                    (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                y += 20
            if rpy_base_box is not None:
                cv2.putText(
                    annotated,
                    f"box rpy (deg): [{rpy_base_box[0]:.1f}, {rpy_base_box[1]:.1f}, {rpy_base_box[2]:.1f}]",
                    (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                y += 20

            cv2.imshow("YOLO Box Pose", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
