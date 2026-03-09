from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import yaml
from cv2 import aruco

MARKER_LENGTH_M = 0.02940
MARKER_SEPARATION_M = 0.0050
MARKERS_X = 5
MARKERS_Y = 7
ARUCO_DICT_ID = aruco.DICT_4X4_50
GRIDBOARD_W_M = MARKERS_X * MARKER_LENGTH_M + (MARKERS_X - 1) * MARKER_SEPARATION_M
GRIDBOARD_H_M = MARKERS_Y * MARKER_LENGTH_M + (MARKERS_Y - 1) * MARKER_SEPARATION_M
DEFAULT_INTRINSICS_PATH = (
    Path(__file__).resolve().parent.parent / "camera_intrinsics.yaml"
)

WIDTH_MARGIN_M = 0.0175 # actual 0.012
HEIGHT_MARGIN_M = 0.0525 #actual 0.022



def load_camera_intrinsics(
    intrinsics_path: Path, camera_name: str
) -> Tuple[np.ndarray, np.ndarray]:
    intrinsics_path = Path(intrinsics_path)
    with intrinsics_path.open("r") as f:
        data = yaml.safe_load(f)
    cameras = data.get("cameras") or {}
    if camera_name not in cameras:
        raise KeyError(f"Camera '{camera_name}' not found in {intrinsics_path}")
    intr = cameras[camera_name]["intrinsics"]
    fx = float(intr["fx"])
    fy = float(intr["fy"])
    cx = float(intr["cx"])
    cy = float(intr["cy"])
    coeffs = np.array(intr.get("coeffs", [0, 0, 0, 0, 0]), dtype=np.float32)
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    return K, coeffs


def _build_board():
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
    return aruco_dict, parameters, board


def detect_board_corners(
    image: np.ndarray, K: np.ndarray, dist_coeffs: np.ndarray
) -> Optional[np.ndarray]:
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    aruco_dict, parameters, board = _build_board()
    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    if ids is None or len(ids) == 0:
        return None
    retval, rvec, tvec = aruco.estimatePoseBoard(
        corners, ids, board, K, dist_coeffs, None, None
    )
    if retval <= 0:
        return None
    board_corners_3d = np.array(
        [
            [0.0, 0.0, 0.0],
            [GRIDBOARD_W_M, 0.0, 0.0],
            [GRIDBOARD_W_M, GRIDBOARD_H_M, 0.0],
            [0.0, GRIDBOARD_H_M, 0.0],
        ],
        dtype=np.float32,
    )
    proj, _ = cv2.projectPoints(board_corners_3d, rvec, tvec, K, dist_coeffs)
    return proj.reshape(-1, 2).astype(np.float32)


def expand_corners(corners: np.ndarray) -> np.ndarray:
    corners = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    p0, p1, p2, p3 = corners
    top_vec = p1 - p0
    bottom_vec = p2 - p3
    left_vec = p3 - p0
    right_vec = p2 - p1
    top_len = float(np.linalg.norm(top_vec))
    bottom_len = float(np.linalg.norm(bottom_vec))
    left_len = float(np.linalg.norm(left_vec))
    right_len = float(np.linalg.norm(right_vec))
    if min(top_len, bottom_len, left_len, right_len) < 1e-6:
        return corners
    top_unit = top_vec / top_len
    bottom_unit = bottom_vec / bottom_len
    left_unit = left_vec / left_len
    right_unit = right_vec / right_len

    top_height_px = (HEIGHT_MARGIN_M * top_len) / GRIDBOARD_H_M
    bottom_height_px = (HEIGHT_MARGIN_M * bottom_len) / GRIDBOARD_H_M
    left_width_px = (WIDTH_MARGIN_M * left_len) / GRIDBOARD_W_M
    right_width_px = (WIDTH_MARGIN_M * right_len) / GRIDBOARD_W_M

    p0e = p0 - top_unit * top_height_px - left_unit * left_width_px
    p1e = p1 + top_unit * top_height_px - right_unit * right_width_px
    p2e = p2 + bottom_unit * bottom_height_px + right_unit * right_width_px
    p3e = p3 - bottom_unit * bottom_height_px + left_unit * left_width_px
    return np.stack([p0e, p1e, p2e, p3e], axis=0)


def _add_point(points: list[np.ndarray], point: np.ndarray, seen: set[tuple]) -> None:
    key = (round(float(point[0]), 3), round(float(point[1]), 3))
    if key in seen:
        return
    seen.add(key)
    points.append(point.astype(np.float32))


def build_prompt_points(corners: np.ndarray) -> np.ndarray:
    corners = expand_corners(corners)
    p0, p1, p2, p3 = corners
    edges = [(p0, p1), (p1, p2), (p2, p3), (p3, p0)]
    edge_lens = [float(np.linalg.norm(b - a)) for a, b in edges]
    if min(edge_lens) < 1e-6:
        raise ValueError("Invalid board corners: edge length too small.")

    points: list[np.ndarray] = []
    seen: set[tuple] = set()

    # Four corners
    for p in corners:
        _add_point(points, p, seen)

    # Split each edge into quarters (add 1/4, 1/2, 3/4)
    for a, b in edges:
        _add_point(points, a + (b - a) * 0.25, seen)
        _add_point(points, a + (b - a) * 0.50, seen)
        _add_point(points, a + (b - a) * 0.75, seen)

    # Diagonals: add 1/4, 1/2, 3/4 points on each diagonal
    diag1 = (p0, p2)
    diag2 = (p1, p3)
    for a, b in (diag1, diag2):
        _add_point(points, a + (b - a) * 0.25, seen)
        _add_point(points, a + (b - a) * 0.50, seen)
        _add_point(points, a + (b - a) * 0.75, seen)

    return np.stack(points, axis=0).astype(np.float32)


class ArucoPromptGenerator:
    def __init__(self, camera_name: str, intrinsics_path: Optional[Path] = None) -> None:
        if intrinsics_path is None:
            intrinsics_path = DEFAULT_INTRINSICS_PATH
        self.K, self.dist_coeffs = load_camera_intrinsics(intrinsics_path, camera_name)
        self._last_points: Optional[np.ndarray] = None
        self._last_corners: Optional[np.ndarray] = None

    def infer(self, image: np.ndarray) -> Optional[np.ndarray]:
        corners = detect_board_corners(image, self.K, self.dist_coeffs)
        if corners is None:
            return None if self._last_points is None else self._last_points.copy()
        points = build_prompt_points(corners)
        self._last_points = points
        self._last_corners = corners
        return points

    def last_corners(self) -> Optional[np.ndarray]:
        return None if self._last_corners is None else self._last_corners.copy()
