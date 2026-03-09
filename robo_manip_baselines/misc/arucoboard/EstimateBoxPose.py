import argparse
from pathlib import Path

import cv2
import numpy as np

from ultralytics import YOLO

from arucoboard_modules.aruco_prompt import (
    GRIDBOARD_H_M,
    GRIDBOARD_W_M,
    load_camera_intrinsics,
)

BOX_DEPTH_M = 0.1140
BASE_T_CAM_PATH = Path(__file__).resolve().parents[2] / "calib" / "base_center_T.calib"
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

    ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    return rvec, tvec


def _draw_axes(frame, rvec, tvec, K, dist_coeffs, axis_len=0.05):
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
    cv2.line(frame, o, x, (0, 0, 255), 2)  # X: red
    cv2.line(frame, o, y, (0, 255, 0), 2)  # Y: green
    cv2.line(frame, o, z, (255, 0, 0), 2)  # Z: blue
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate board pose via PnP from YOLO keypoints.")
    parser.add_argument("pt_path", help="Path to .pt model file.")
    parser.add_argument("video_path", help="Path to video file.")
    args = parser.parse_args()

    calib_path = Path(__file__).resolve().parent / "camera_intrinsics.yaml"
    K, dist_coeffs = load_camera_intrinsics(calib_path, "front")
    base_T_cam = np.loadtxt(BASE_T_CAM_PATH).astype(np.float32)

    model = YOLO(args.pt_path)
    cap = cv2.VideoCapture(args.video_path)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        results = model.track(frame, persist=True)
        annotated_frame = results[0].plot()

        keypoints = results[0].keypoints
        if keypoints is not None and keypoints.xy is not None and len(keypoints.xy) > 0:
            pts = keypoints.xy[0].detach().cpu().numpy()
            pnp = _solve_pnp_from_keypoints(pts, K, dist_coeffs)
            if pnp is not None:
                rvec, tvec = pnp
                annotated_frame = _draw_axes(annotated_frame, rvec, tvec, K, dist_coeffs)

                R_board, _ = cv2.Rodrigues(rvec)
                R_flip_x = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32)
                center_offset = np.array(
                    [GRIDBOARD_W_M * 0.5, GRIDBOARD_H_M * 0.5, 0.0], dtype=np.float32
                )
                z_offset = np.array([0.0, 0.0, -BOX_DEPTH_M * 0.5], dtype=np.float32)

                t_board_box = center_offset + R_flip_x @ z_offset
                R_cam_box = R_board @ R_flip_x
                t_cam_box = R_board @ t_board_box.reshape(3, 1) + tvec.reshape(3, 1)

                cam_T_box = np.eye(4, dtype=np.float32)
                cam_T_box[:3, :3] = R_cam_box.astype(np.float32)
                cam_T_box[:3, 3] = t_cam_box.flatten().astype(np.float32)
                base_T_box = base_T_cam @ cam_T_box
                print(f"t_base_box: {base_T_box[:3, 3]}")

                rvec_box, _ = cv2.Rodrigues(R_cam_box)
                annotated_frame = _draw_axes(
                    annotated_frame, rvec_box, t_cam_box, K, dist_coeffs
                )

        cv2.imshow("YOLO PnP", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
