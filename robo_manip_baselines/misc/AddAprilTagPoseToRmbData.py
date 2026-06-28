import argparse
import os

import cv2
import numpy as np
from tqdm import tqdm

from robo_manip_baselines.common import (
    DataKey,
    RmbData,
    find_rmb_files,
)


def parse_argument():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "path",
        type=str,
        help="path to data (*.hdf5 or *.rmb) or directory containing them",
    )
    parser.add_argument(
        "--camera_name",
        type=str,
        default="front",
        help="name of the RGB camera used for AprilTag detection",
    )
    parser.add_argument(
        "--tag_size",
        type=float,
        default=0.03385,
        help="AprilTag side length [m]",
    )
    parser.add_argument(
        "--tag_id",
        type=int,
        default=None,
        help="AprilTag id to use; if omitted, the largest detected tag is used",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="whether to overwrite existing values if they exist",
    )
    parser.add_argument(
        "--save_video",
        action="store_true",
        help="whether to save an mp4 video with AprilTag detection overlays",
    )
    parser.add_argument(
        "--video_fps",
        type=float,
        default=None,
        help="fps of the output mp4 video; if omitted, it is inferred from the RMB time data",
    )

    return parser.parse_args()


class AddAprilTagPoseToRmbData:
    TAG_FAMILY = "tag25h9"

    def __init__(
        self,
        path,
        camera_name="front",
        tag_size=0.03385,
        tag_id=None,
        overwrite=False,
        save_video=False,
        video_fps=None,
    ):
        self.path = path
        self.camera_name = camera_name
        self.tag_size = tag_size
        self.tag_id = tag_id
        self.overwrite = overwrite
        self.save_video = save_video
        self.video_fps = video_fps

        self.rgb_key = DataKey.get_rgb_image_key(camera_name)
        self.pose_key = f"{camera_name}_apriltag_pose_xy_axis"
        self.corners_key = f"{camera_name}_apriltag_corners"
        self.detected_key = f"{camera_name}_apriltag_detected"
        self.id_key = f"{camera_name}_apriltag_id"

    def run(self):
        self.setup_detector()
        rmb_path_list = find_rmb_files(self.path)
        print(
            f"[{self.__class__.__name__}] Add '{self.pose_key}' from '{self.rgb_key}'."
        )
        print(f"  - tag family: {self.TAG_FAMILY}")
        print(f"  - tag size: {self.tag_size} m")
        print(f"  - tag id: {self.tag_id}")

        for rmb_path in tqdm(rmb_path_list):
            tqdm.write(f"[{self.__class__.__name__}] Open {rmb_path}")
            with RmbData(rmb_path, mode="r+") as rmb_data:
                self.assert_output_keys(rmb_data, rmb_path)
                if self.rgb_key not in rmb_data.keys():
                    raise KeyError(
                        f"[{self.__class__.__name__}] '{self.rgb_key}' is not found: {rmb_path}"
                    )

                rgb_image_seq = np.asarray(rmb_data[self.rgb_key][:])
                camera_matrix = self.build_camera_matrix(rmb_data, rgb_image_seq)
                poses, corners, detected, tag_ids, rvecs, tvecs = self.detect_sequence(
                    rmb_path,
                    rgb_image_seq,
                    camera_matrix,
                )
                self.save(rmb_data, poses, corners, detected, tag_ids)
                if self.save_video:
                    self.save_detection_video(
                        rmb_path,
                        rmb_data,
                        rgb_image_seq,
                        corners,
                        detected,
                        tag_ids,
                        camera_matrix,
                        rvecs,
                        tvecs,
                    )

    def setup_detector(self):
        dictionary = cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_APRILTAG_25h9
        )
        self.detector = cv2.aruco.ArucoDetector(
            dictionary,
            self.build_detector_parameters(cv2.aruco.CORNER_REFINE_APRILTAG),
        )

    def build_detector_parameters(self, corner_refinement_method):
        parameters = cv2.aruco.DetectorParameters()
        parameters.cornerRefinementMethod = corner_refinement_method
        parameters.adaptiveThreshWinSizeMax = 101
        parameters.adaptiveThreshWinSizeStep = 2
        parameters.adaptiveThreshConstant = 3
        parameters.minMarkerPerimeterRate = 0.005
        parameters.maxMarkerPerimeterRate = 10.0
        parameters.minCornerDistanceRate = 0.01
        parameters.minDistanceToBorder = 0
        parameters.errorCorrectionRate = 1.0
        parameters.maxErroneousBitsInBorderRate = 0.8
        parameters.aprilTagMinClusterPixels = 1
        parameters.aprilTagMinWhiteBlackDiff = 1
        parameters.aprilTagMaxLineFitMse = 50.0
        parameters.aprilTagCriticalRad = 0.05
        parameters.polygonalApproxAccuracyRate = 0.08
        parameters.detectInvertedMarker = True
        return parameters

    def assert_output_keys(self, rmb_data, rmb_path):
        output_keys = [
            self.pose_key,
            self.corners_key,
            self.detected_key,
            self.id_key,
        ]
        for key in output_keys:
            if key not in rmb_data.keys():
                continue
            if self.overwrite:
                del rmb_data.h5file[key]
            else:
                raise ValueError(
                    f"[{self.__class__.__name__}] '{key}' already exists: "
                    f"{rmb_path} (use --overwrite to replace)"
                )

    def build_camera_matrix(self, rmb_data, rgb_image_seq):
        image_height, image_width = rgb_image_seq.shape[1:3]
        fovy = self.get_camera_fovy(rmb_data)
        fy = image_height / (2.0 * np.tan(np.deg2rad(fovy) / 2.0))
        fx = fy
        cx = (image_width - 1.0) / 2.0
        cy = (image_height - 1.0) / 2.0
        return np.array(
            [
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    def get_camera_fovy(self, rmb_data):
        rgb_fovy_key = self.rgb_key + "_fovy"
        depth_fovy_key = DataKey.get_depth_image_key(self.camera_name) + "_fovy"
        if rgb_fovy_key in rmb_data.attrs:
            return rmb_data.attrs[rgb_fovy_key]
        if depth_fovy_key in rmb_data.attrs:
            return rmb_data.attrs[depth_fovy_key]
        raise KeyError(
            f"[{self.__class__.__name__}] Camera fovy attr is not found: "
            f"'{rgb_fovy_key}' or '{depth_fovy_key}'"
        )

    def detect_sequence(self, rmb_path, rgb_image_seq, camera_matrix):
        poses = np.full((len(rgb_image_seq), 9), np.nan, dtype=np.float64)
        corners = np.full((len(rgb_image_seq), 4, 2), np.nan, dtype=np.float64)
        detected = np.zeros(len(rgb_image_seq), dtype=np.bool_)
        tag_ids = np.full(len(rgb_image_seq), -1, dtype=np.int32)
        rvecs = np.full((len(rgb_image_seq), 3), np.nan, dtype=np.float64)
        tvecs = np.full((len(rgb_image_seq), 3), np.nan, dtype=np.float64)
        previous_rotation_matrix = None
        previous_corners = None

        for frame_idx, rgb_image in enumerate(rgb_image_seq):
            detection = self.detect_single(
                rgb_image,
                camera_matrix,
                previous_rotation_matrix,
                previous_corners,
            )
            if detection is None:
                continue
            poses[frame_idx] = detection["pose"]
            corners[frame_idx] = detection["corners"]
            detected[frame_idx] = True
            tag_ids[frame_idx] = detection["tag_id"]
            rvecs[frame_idx] = detection["rvec"]
            tvecs[frame_idx] = detection["tvec"]
            previous_rotation_matrix = detection["rotation_matrix"]
            previous_corners = detection["corners"]

        if not np.any(detected):
            raise ValueError(
                f"[{self.__class__.__name__}] No AprilTag was detected from "
                f"'{self.rgb_key}': {rmb_path}"
            )

        self.fill_missing_with_previous(
            rmb_path,
            poses,
            corners,
            tag_ids,
            rvecs,
            tvecs,
            detected,
        )
        return poses, corners, detected, tag_ids, rvecs, tvecs

    def detect_single(
        self,
        rgb_image,
        camera_matrix,
        previous_rotation_matrix=None,
        previous_corners=None,
    ):
        gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        corners_list, ids = self.detect_markers_with_fallback(
            gray_image,
            previous_corners,
        )
        if ids is None:
            return None

        selected_idx = self.select_detection(corners_list, ids)
        if selected_idx is None:
            return None

        image_points = corners_list[selected_idx].reshape(4, 2).astype(np.float64)
        tag_id = int(ids[selected_idx, 0])
        pose_xy_axis, rvec, tvec, rotation_matrix = self.estimate_pose(
            image_points,
            camera_matrix,
            previous_rotation_matrix,
        )
        return {
            "pose": pose_xy_axis,
            "corners": image_points,
            "tag_id": tag_id,
            "rvec": rvec,
            "tvec": tvec,
            "rotation_matrix": rotation_matrix,
        }

    def detect_markers_with_fallback(self, gray_image, previous_corners=None):
        for target_gray_image, scale in self.get_gray_image_fallbacks(gray_image):
            corners_list, ids, _rejected = self.detector.detectMarkers(
                target_gray_image
            )
            if not self.has_target_tag_id(ids):
                continue
            if scale != 1:
                corners_list = [corners / scale for corners in corners_list]
            return corners_list, ids
        if previous_corners is None:
            return None, None

        crop_info = self.crop_around_previous_corners(gray_image, previous_corners)
        if crop_info is None:
            return None, None
        crop_gray_image, offset = crop_info
        for target_gray_image, scale in self.get_crop_gray_image_fallbacks(
            crop_gray_image
        ):
            corners_list, ids, _rejected = self.detector.detectMarkers(
                target_gray_image
            )
            if not self.has_target_tag_id(ids):
                continue
            corners_list = [corners / scale + offset for corners in corners_list]
            return corners_list, ids
        return None, None

    def has_target_tag_id(self, ids):
        if ids is None:
            return False
        if self.tag_id is None:
            return True
        return self.tag_id in ids[:, 0]

    def get_gray_image_fallbacks(self, gray_image):
        clahe2 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe3 = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        sharp_gray_image = self.sharpen_gray_image(gray_image)
        gray_images = [
            gray_image,
            cv2.equalizeHist(gray_image),
            clahe2.apply(gray_image),
            clahe3.apply(gray_image),
            sharp_gray_image,
        ]

        for target_gray_image in gray_images:
            yield target_gray_image, 1
        for target_gray_image in [
            gray_image,
            sharp_gray_image,
            clahe2.apply(gray_image),
        ]:
            scale = 2
            yield (
                cv2.resize(
                    target_gray_image,
                    None,
                    fx=scale,
                    fy=scale,
                    interpolation=cv2.INTER_CUBIC,
                ),
                scale,
            )

    def crop_around_previous_corners(self, gray_image, previous_corners):
        points = previous_corners.reshape(4, 2)
        min_xy = points.min(axis=0)
        max_xy = points.max(axis=0)
        tag_size_px = float(np.max(max_xy - min_xy))
        margin = max(80.0, 3.0 * tag_size_px)
        image_height, image_width = gray_image.shape[:2]
        x0 = max(int(np.floor(min_xy[0] - margin)), 0)
        y0 = max(int(np.floor(min_xy[1] - margin)), 0)
        x1 = min(int(np.ceil(max_xy[0] + margin)), image_width)
        y1 = min(int(np.ceil(max_xy[1] + margin)), image_height)
        if x1 <= x0 or y1 <= y0:
            return None
        return gray_image[y0:y1, x0:x1], np.array([x0, y0], dtype=np.float32)

    def get_crop_gray_image_fallbacks(self, gray_image):
        for target_gray_image in [
            gray_image,
            self.sharpen_gray_image(gray_image),
        ]:
            for scale in (2, 3):
                yield (
                    cv2.resize(
                        target_gray_image,
                        None,
                        fx=scale,
                        fy=scale,
                        interpolation=cv2.INTER_CUBIC,
                    ),
                    scale,
                )

    def sharpen_gray_image(self, gray_image):
        blurred_gray_image = cv2.GaussianBlur(gray_image, (0, 0), 1.0)
        return cv2.addWeighted(gray_image, 1.8, blurred_gray_image, -0.8, 0)

    def select_detection(self, corners_list, ids):
        if self.tag_id is not None:
            matched = np.where(ids[:, 0] == self.tag_id)[0]
            if len(matched) == 0:
                return None
            assert len(matched) == 1, matched
            return int(matched[0])

        areas = [
            cv2.contourArea(corners.reshape(4, 2).astype(np.float32))
            for corners in corners_list
        ]
        return int(np.argmax(areas))

    def estimate_pose(self, image_points, camera_matrix, previous_rotation_matrix):
        half_size = self.tag_size / 2.0
        object_points = np.array(
            [
                [-half_size, half_size, 0.0],
                [half_size, half_size, 0.0],
                [half_size, -half_size, 0.0],
                [-half_size, -half_size, 0.0],
            ],
            dtype=np.float64,
        )
        success, rvecs, tvecs, reprojection_errors = cv2.solvePnPGeneric(
            object_points,
            image_points,
            camera_matrix,
            np.zeros(5),
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        assert success
        rvec, tvec, rotation_matrix = self.select_pose_solution(
            rvecs,
            tvecs,
            reprojection_errors,
            previous_rotation_matrix,
        )
        return (
            self.get_pose_xy_axis(rotation_matrix, tvec),
            rvec,
            tvec,
            rotation_matrix,
        )

    def select_pose_solution(
        self,
        rvecs,
        tvecs,
        reprojection_errors,
        previous_rotation_matrix,
    ):
        if previous_rotation_matrix is None:
            selected_idx = int(np.argmin(np.asarray(reprojection_errors).reshape(-1)))
        else:
            rotation_scores = []
            for rvec in rvecs:
                rotation_matrix, _jacobian = cv2.Rodrigues(rvec)
                rotation_scores.append(
                    np.trace(previous_rotation_matrix.T @ rotation_matrix)
                )
            selected_idx = int(np.argmax(rotation_scores))

        rvec = rvecs[selected_idx].reshape(3)
        tvec = tvecs[selected_idx].reshape(3)
        rotation_matrix, _jacobian = cv2.Rodrigues(rvec)
        return rvec, tvec, rotation_matrix

    def get_pose_xy_axis(self, rotation_matrix, position):
        return np.concatenate(
            [
                position,
                rotation_matrix[:, 0],
                rotation_matrix[:, 1],
            ]
        )

    def fill_missing_with_previous(
        self,
        rmb_path,
        poses,
        corners,
        tag_ids,
        rvecs,
        tvecs,
        detected,
    ):
        if not detected[0]:
            raise ValueError(
                f"[{self.__class__.__name__}] AprilTag was not detected at frame 0 "
                f"from '{self.rgb_key}'. Frame 0 detection is required because missing "
                f"frames are filled with the previous frame: {rmb_path}"
            )

        for frame_idx in range(1, len(detected)):
            if detected[frame_idx]:
                continue
            poses[frame_idx] = poses[frame_idx - 1]
            corners[frame_idx] = corners[frame_idx - 1]
            tag_ids[frame_idx] = tag_ids[frame_idx - 1]
            rvecs[frame_idx] = rvecs[frame_idx - 1]
            tvecs[frame_idx] = tvecs[frame_idx - 1]

    def save_detection_video(
        self,
        rmb_path,
        rmb_data,
        rgb_image_seq,
        corners,
        detected,
        tag_ids,
        camera_matrix,
        rvecs,
        tvecs,
    ):
        output_path = self.get_video_output_path(rmb_path)
        fps = self.get_video_fps(rmb_data)
        height, width = rgb_image_seq.shape[1:3]
        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(
                f"[{self.__class__.__name__}] Failed to open video writer: {output_path}"
            )

        for frame_idx, rgb_image in enumerate(rgb_image_seq):
            frame = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            self.draw_detection_overlay(
                frame,
                frame_idx,
                corners[frame_idx],
                detected[frame_idx],
                tag_ids[frame_idx],
                camera_matrix,
                rvecs[frame_idx],
                tvecs[frame_idx],
            )
            writer.write(frame)
        writer.release()
        tqdm.write(f"[{self.__class__.__name__}] Save video {output_path}")

    def get_video_output_path(self, rmb_path):
        output_filename = f"{self.pose_key}_detection.mp4"
        if rmb_path.rstrip("/").endswith(".rmb"):
            return os.path.join(rmb_path, output_filename)

        base_path, _ext = os.path.splitext(rmb_path)
        return base_path + "_" + output_filename

    def get_video_fps(self, rmb_data):
        if self.video_fps is not None:
            return self.video_fps
        if DataKey.TIME not in rmb_data.keys():
            return 30.0

        time = np.asarray(rmb_data[DataKey.TIME][:], dtype=np.float64)
        if len(time) < 2:
            return 30.0
        dt = np.median(np.diff(time))
        if dt <= 0:
            return 30.0
        return float(1.0 / dt)

    def draw_detection_overlay(
        self,
        frame,
        frame_idx,
        corners,
        detected,
        tag_id,
        camera_matrix,
        rvec,
        tvec,
    ):
        color = (0, 255, 0) if detected else (0, 255, 255)
        points = np.round(corners).astype(np.int32)
        cv2.polylines(frame, [points], isClosed=True, color=color, thickness=2)
        for point_idx, point in enumerate(points):
            cv2.circle(frame, tuple(point), 4, color, -1)
            cv2.putText(
                frame,
                str(point_idx),
                tuple(point + np.array([4, -4])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )

        cv2.drawFrameAxes(
            frame,
            camera_matrix,
            np.zeros(5),
            rvec,
            tvec,
            self.tag_size * 0.5,
        )
        status = "detected" if detected else "filled_previous"
        cv2.putText(
            frame,
            f"frame={frame_idx} tag_id={tag_id} {status}",
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )

    def save(self, rmb_data, poses, corners, detected, tag_ids):
        rmb_data.h5file[self.pose_key] = poses
        rmb_data.h5file[self.corners_key] = corners
        rmb_data.h5file[self.detected_key] = detected
        rmb_data.h5file[self.id_key] = tag_ids

        rmb_data.attrs[self.pose_key + "_source_key"] = self.rgb_key
        rmb_data.attrs[self.pose_key + "_tag_family"] = self.TAG_FAMILY
        rmb_data.attrs[self.pose_key + "_tag_size"] = self.tag_size
        rmb_data.attrs[self.pose_key + "_tag_id"] = (
            -1 if self.tag_id is None else self.tag_id
        )
        rmb_data.attrs[self.pose_key + "_fill_missing"] = (
            "previous_frame; frame0_detection_required"
        )
        rmb_data.attrs[self.pose_key + "_format"] = (
            "tx ty tz x_axis_x x_axis_y x_axis_z y_axis_x y_axis_y y_axis_z"
        )


if __name__ == "__main__":
    add_apriltag_pose = AddAprilTagPoseToRmbData(**vars(parse_argument()))
    add_apriltag_pose.run()
