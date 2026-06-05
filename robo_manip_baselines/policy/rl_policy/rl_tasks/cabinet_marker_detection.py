from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from cv2 import aruco

BIG_MARKER_LENGTH_M = 0.02940
BIG_MARKER_SEPARATION_M = 0.0050
BIG_MARKERS_X = 5
BIG_MARKERS_Y = 7
BIG_ARUCO_DICT_ID = aruco.DICT_4X4_50
BIG_PANEL_Z_OFFSET_M = 0.003
BIG_BOARD_RZ_OFFSET_RAD = np.deg2rad(90.0)

SMALL_BOARD_DICT_ID = aruco.DICT_5X5_250
SMALL_BOARD_MARKER_IDS = [100, 101, 102]
SMALL_BOARD_MARKER_LENGTH_M = 0.0287
SMALL_BOARD_MARKER_GAP_M = 0.005
SMALL_BOARD_W_M = SMALL_BOARD_MARKER_LENGTH_M * len(
    SMALL_BOARD_MARKER_IDS
) + SMALL_BOARD_MARKER_GAP_M * (len(SMALL_BOARD_MARKER_IDS) - 1)
SMALL_BOARD_H_M = SMALL_BOARD_MARKER_LENGTH_M
SMALL_BOARD_TARGET_Y_OFFSET_M = -0.0825
SMALL_BOARD_TARGET_RY_OFFSET_RAD = np.deg2rad(-90.0)

BASE_CENTER_T_PATH = Path("robo_manip_baselines/calib/base_center_T.calib")
RES_W, RES_H, FPS = 1920, 1080, 30
USE_SERIAL = "314422070401"


def rotation_z(rad: float) -> np.ndarray:
    c = np.cos(rad)
    s = np.sin(rad)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def rotation_y(rad: float) -> np.ndarray:
    c = np.cos(rad)
    s = np.sin(rad)
    return np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=np.float32,
    )


def get_aruco_dictionary(dict_id):
    try:
        return aruco.getPredefinedDictionary(dict_id)
    except AttributeError:
        return aruco.Dictionary_get(dict_id)


def get_aruco_parameters():
    try:
        return aruco.DetectorParameters_create()
    except AttributeError:
        return aruco.DetectorParameters()


def detect_markers(gray, aruco_dict, aruco_params):
    if hasattr(aruco, "ArucoDetector"):
        detector = aruco.ArucoDetector(aruco_dict, aruco_params)
        return detector.detectMarkers(gray)
    return aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)


def build_small_board(small_board_dict):
    return aruco.GridBoard(
        (len(SMALL_BOARD_MARKER_IDS), 1),
        SMALL_BOARD_MARKER_LENGTH_M,
        SMALL_BOARD_MARKER_GAP_M,
        small_board_dict,
        np.array(SMALL_BOARD_MARKER_IDS, dtype=np.int32),
    )


def estimate_small_board_pose(corners, ids, board, K, dist_coeffs):
    if ids is None or len(ids) == 0:
        return None, None

    retval, rvec, tvec = aruco.estimatePoseBoard(
        corners, ids, board, K, dist_coeffs, None, None
    )
    if retval <= 0:
        return None, None

    R_board, _ = cv2.Rodrigues(rvec)
    center_offset = np.array(
        [SMALL_BOARD_W_M * 0.5, SMALL_BOARD_H_M * 0.5, 0.0], dtype=np.float32
    )
    t_center = R_board @ center_offset.reshape(3, 1) + tvec.reshape(3, 1)
    target_offset = np.array(
        [-0.00, SMALL_BOARD_TARGET_Y_OFFSET_M, 0.0], dtype=np.float32
    )
    R_target = R_board @ rotation_y(SMALL_BOARD_TARGET_RY_OFFSET_RAD)
    t_target = R_board @ target_offset.reshape(3, 1) + t_center.reshape(3, 1)
    rvec_target, _ = cv2.Rodrigues(R_target)
    return rvec_target.reshape(3, 1), t_target.reshape(3, 1)
