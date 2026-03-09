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


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize YOLO tracking on a video.")
    parser.add_argument("pt_path", help="Path to .pt model file.")
    parser.add_argument("video_path", help="Path to video file.")
    args = parser.parse_args()

    # Load the YOLO model
    model = YOLO(args.pt_path)

    # Open the video file
    cap = cv2.VideoCapture(args.video_path)

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLO tracking on the frame, persisting tracks between frames
            results = model.track(frame, persist=True)
            #print(f"keypoints results[0]:{results[0].keypoints}")
            # Visualize the results on the frame
            
            annotated_frame = results[0].plot()

            annotated_frame = draw_topk_keypoints(
                annotated_frame,
                results[0].keypoints,
                topk=3,
                color=(0, 0, 255),
                radius=6,
                thickness=2,
            )


            # Display the annotated frame
            cv2.imshow("YOLO Tracking", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
