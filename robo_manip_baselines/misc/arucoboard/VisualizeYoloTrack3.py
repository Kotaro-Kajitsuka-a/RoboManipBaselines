import argparse

import cv2

from ultralytics import YOLO


def draw_topk_keypoints(frame, keypoints, topk=3, color=(0, 0, 255), radius=6, thickness=2):
    if keypoints is None or keypoints.conf is None:
        return frame
    xy = keypoints.xy
    conf = keypoints.conf
    if xy is None or conf is None or len(xy) == 0:
        return frame
    pts = xy[0].detach().cpu().numpy()
    conf_vals = conf[0].detach().cpu().numpy()
    k = min(int(topk), len(conf_vals))
    if k <= 0:
        return frame
    top_idx = conf_vals.argsort()[-k:][::-1]
    for idx in top_idx:
        x, y = pts[idx]
        cv2.circle(
            frame,
            (int(round(x)), int(round(y))),
            int(radius),
            color,
            int(thickness),
        )
    return frame


def draw_f1_f2_axes_2d(frame, keypoints, color_x=(0, 0, 255), color_y=(0, 255, 0), thickness=2):
    if keypoints is None or keypoints.xy is None or len(keypoints.xy) == 0:
        return frame
    pts = keypoints.xy[0].detach().cpu().numpy()
    if pts.shape[0] < 3:
        return frame
    p0 = pts[0]
    p1 = pts[1]
    p3 = pts[3]
    o = (int(round(p0[0])), int(round(p0[1])))
    pxi = (int(round(p1[0])), int(round(p1[1])))
    pyi = (int(round(p3[0])), int(round(p3[1])))
    cv2.line(frame, o, pxi, color_x, thickness)
    cv2.line(frame, o, pyi, color_y, thickness)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize YOLO tracking on a video.")
    parser.add_argument("pt_path", help="Path to .pt model file.")
    parser.add_argument("video_path", help="Path to video file.")
    args = parser.parse_args()

    model = YOLO(args.pt_path)
    cap = cv2.VideoCapture(args.video_path)

    while cap.isOpened():
        success, frame = cap.read()

        if success:
            results = model.track(frame, persist=True)
            annotated_frame = results[0].plot()

            annotated_frame = draw_topk_keypoints(
                annotated_frame,
                results[0].keypoints,
                topk=3,
                color=(0, 0, 255),
                radius=6,
                thickness=2,
            )
            annotated_frame = draw_f1_f2_axes_2d(
                annotated_frame,
                results[0].keypoints,
                color_x=(0, 0, 255),
                color_y=(0, 255, 0),
                thickness=2,
            )

            cv2.imshow("YOLO Tracking", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
