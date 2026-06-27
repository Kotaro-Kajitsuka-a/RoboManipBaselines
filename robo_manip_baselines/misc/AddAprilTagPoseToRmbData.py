import argparse

import cv2
import numpy as np
from tqdm import tqdm

from robo_manip_baselines.common import (
    DataKey,
    RmbData,
    find_rmb_files,
    get_pose_from_rot_pos,
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
    ):
        self.path = path
        self.camera_name = camera_name
        self.tag_size = tag_size
        self.tag_id = tag_id
        self.overwrite = overwrite

        self.rgb_key = DataKey.get_rgb_image_key(camera_name)
        self.pose_key = f"{camera_name}_apriltag_pose"
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
                poses, corners, detected, tag_ids = self.detect_sequence(
                    rgb_image_seq,
                    camera_matrix,
                )
                self.save(rmb_data, poses, corners, detected, tag_ids)

    def setup_detector(self):
        dictionary = cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_APRILTAG_25h9
        )
        parameters = cv2.aruco.DetectorParameters()
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
        self.detector = cv2.aruco.ArucoDetector(dictionary, parameters)

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

    def detect_sequence(self, rgb_image_seq, camera_matrix):
        poses = np.full((len(rgb_image_seq), 7), np.nan, dtype=np.float64)
        corners = np.full((len(rgb_image_seq), 4, 2), np.nan, dtype=np.float64)
        detected = np.zeros(len(rgb_image_seq), dtype=np.bool_)
        tag_ids = np.full(len(rgb_image_seq), -1, dtype=np.int32)

        for frame_idx, rgb_image in enumerate(rgb_image_seq):
            detection = self.detect_single(rgb_image, camera_matrix)
            if detection is None:
                continue
            poses[frame_idx] = detection["pose"]
            corners[frame_idx] = detection["corners"]
            detected[frame_idx] = True
            tag_ids[frame_idx] = detection["tag_id"]

        if not np.any(detected):
            raise ValueError(
                f"[{self.__class__.__name__}] No AprilTag was detected from '{self.rgb_key}'."
            )

        self.fill_missing_with_previous(poses, corners, tag_ids, detected)
        return poses, corners, detected, tag_ids

    def detect_single(self, rgb_image, camera_matrix):
        gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        corners_list, ids, _rejected = self.detector.detectMarkers(gray_image)
        if ids is None:
            return None

        selected_idx = self.select_detection(corners_list, ids)
        if selected_idx is None:
            return None

        image_points = corners_list[selected_idx].reshape(4, 2).astype(np.float64)
        tag_id = int(ids[selected_idx, 0])
        pose = self.estimate_pose(image_points, camera_matrix)
        return {
            "pose": pose,
            "corners": image_points,
            "tag_id": tag_id,
        }

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

    def estimate_pose(self, image_points, camera_matrix):
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
        success, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            camera_matrix,
            np.zeros(5),
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        assert success
        rotation_matrix, _jacobian = cv2.Rodrigues(rvec)
        return get_pose_from_rot_pos(rotation_matrix, tvec.reshape(3))

    def fill_missing_with_previous(self, poses, corners, tag_ids, detected):
        first_detected_idx = int(np.where(detected)[0][0])
        poses[:first_detected_idx] = poses[first_detected_idx]
        corners[:first_detected_idx] = corners[first_detected_idx]
        tag_ids[:first_detected_idx] = tag_ids[first_detected_idx]

        for frame_idx in range(first_detected_idx + 1, len(detected)):
            if detected[frame_idx]:
                continue
            poses[frame_idx] = poses[frame_idx - 1]
            corners[frame_idx] = corners[frame_idx - 1]
            tag_ids[frame_idx] = tag_ids[frame_idx - 1]

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
            "previous_frame; leading_missing_frames_use_first_detection"
        )
        rmb_data.attrs[self.pose_key + "_format"] = "tx ty tz qw qx qy qz"


if __name__ == "__main__":
    add_apriltag_pose = AddAprilTagPoseToRmbData(**vars(parse_argument()))
    add_apriltag_pose.run()
