#camera→boxの検出用

import cv2
import time
import numpy as np
import pyrealsense2 as rs
from cv2 import aruco

# ==== ユーザー設定 ============================================================
MARKER_LENGTH_M      = 0.02940       # 2.940 cm
MARKER_SEPARATION_M  = 0.0050        # 0.50 cm
# GridBoard サイズ (markersX = 横のマーカー数, markersY = 縦のマーカー数)
MARKERS_X = 5   # 列数
MARKERS_Y = 7   # 行数
# 使用した ArUco 辞書（※印刷時に使ったものと必ず一致させる）
ARUCO_DICT_ID = aruco.DICT_4X4_50
# ボックスの高さ（奥行き）: 112.5 mm -> 中心まで 56.25 mm
BOX_DEPTH_M = 0.1140
# RealSense 設定
RES_W, RES_H, FPS = 1920, 1080, 30
USE_SERIAL = "314422070401"  # 必要に応じてカメラシリアル差し替え
# ==== ユーティリティ関数 ======================================================
def rotmat_to_rpy_deg(R: np.ndarray):
    """回転行列 -> RPY (deg)"""
    sy = -R[2, 0]
    sy = np.clip(sy, -1.0, 1.0)
    pitch = np.arcsin(sy)
    if abs(sy) < 0.999999:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw  = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        yaw  = 0.0
    return np.degrees([roll, pitch, yaw])
def put_text_safe(img, text, x, y, scale=0.55, thick=2):
    """画像外にはみ出さないようにテキスト描画"""
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x = int(np.clip(x, 0, img.shape[1] - tw - 2))
    y = int(np.clip(y, th + 2, img.shape[0] - 2))
    cv2.putText(img, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, scale,
                (255, 255, 255), thick, cv2.LINE_AA)
def draw_axes(img, K, rvec, tvec, axis_len=0.05, thickness=2):
    """カメラ座標系での3軸を画像に描画"""
    origin = np.float32([[0, 0, 0]])
    axes   = np.float32([[axis_len, 0, 0],
                         [0, axis_len, 0],
                         [0, 0, axis_len]])
    pts3d  = np.vstack([origin, axes]).reshape(-1, 1, 3)
    proj, _ = cv2.projectPoints(pts3d, rvec, tvec, K, None)
    p0, pX, pY, pZ = [tuple(p.ravel().astype(int)) for p in proj]
    cv2.line(img, p0, pX, (0,   0, 255), thickness)  # X: 赤
    cv2.line(img, p0, pY, (0, 255,   0), thickness)  # Y: 緑
    cv2.line(img, p0, pZ, (255, 0,   0), thickness)  # Z: 青
# ==== ArUco Board 定義 ========================================================
# 辞書取得
try:
    aruco_dict = aruco.getPredefinedDictionary(ARUCO_DICT_ID)
except AttributeError:
    aruco_dict = aruco.Dictionary_get(ARUCO_DICT_ID)
# Detector パラメータ
try:
    parameters = aruco.DetectorParameters_create()
except AttributeError:
    parameters = aruco.DetectorParameters()
# ★ ここが重要：GridBoard は「コンストラクタ」で作る（OpenCV 4.12 スタイル）
#   size は (markersX, markersY) のタプル
board = aruco.GridBoard(
    (MARKERS_X, MARKERS_Y),
    MARKER_LENGTH_M,
    MARKER_SEPARATION_M,
    aruco_dict
)
# Board の物理サイズ（原点は左上。X 右, Y 下, Z は表面法線）
BOARD_W_M = MARKERS_X * MARKER_LENGTH_M + (MARKERS_X - 1) * MARKER_SEPARATION_M
BOARD_H_M = MARKERS_Y * MARKER_LENGTH_M + (MARKERS_Y - 1) * MARKER_SEPARATION_M


# ==== RealSense 初期化 ========================================================
pipeline = rs.pipeline()
config = rs.config()
config.enable_device(USE_SERIAL)
config.enable_stream(rs.stream.color, RES_W, RES_H, rs.format.bgr8, FPS)
profile = pipeline.start(config)
# 露出固定（必要に応じてコメントアウト）
sensor = profile.get_device().first_color_sensor()
try:
    sensor.set_option(rs.option.enable_auto_exposure, 0)
    sensor.set_option(rs.option.exposure, 140)  # 環境光で調整
    sensor.set_option(rs.option.gain, 64)
except Exception:
    pass
# カメラ内部パラメータ取得
color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
intr = color_stream.get_intrinsics()
fx, fy, cx, cy = intr.fx, intr.fy, intr.ppx, intr.ppy
K = np.array([[fx, 0,  cx],
              [0,  fy, cy],
              [0,  0,  1]], dtype=np.float32)
dist_coeffs = np.array(intr.coeffs[:5], dtype=np.float32)
print("Camera intrinsics:")
print(" fx, fy =", fx, fy)
print(" cx, cy =", cx, cy)
print(" dist   =", dist_coeffs)
# ==== メインループ ============================================================
t_prev, fps = time.time(), 0.0
last_print = 0.0
try:
    while True:
        frames = pipeline.wait_for_frames()
        cf = frames.get_color_frame()
        if not cf:
            continue
        img = np.asanyarray(cf.get_data())
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # ---- マーカー検出 ----
        corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        # board pose 推定用
        board_pose_ok = False
        box_pose_ok = False
        rvec_board = None
        tvec_board = None
        rpy_board  = None
        dist_board = None
        rvec_box   = None
        tvec_box   = None
        box_center_cam = None
        used_ids   = []
        if ids is not None and len(ids) > 0:
            used_ids = ids.flatten().tolist()
            # 見つかったマーカーを描画（デバッグ用）
            aruco.drawDetectedMarkers(img, corners, ids)
            # ---- board 全体の姿勢推定 ----
            # retval > 0 のとき、rvec/tvec に board (GridBoard) の姿勢が入る
            retval, rvec, tvec = aruco.estimatePoseBoard(
                corners, ids, board, K, dist_coeffs, None, None
            )
            if retval > 0:
                board_pose_ok = True
                rvec_board, tvec_board = rvec, tvec
                R_board, _ = cv2.Rodrigues(rvec_board)
                rpy_board = rotmat_to_rpy_deg(R_board)
                dist_board = float(np.linalg.norm(tvec_board))
                # board 原点（GridBoard の定義座標系）に座標軸を描画
                # 軸長はボードサイズの 60% くらいにしてみる
                board_size_max = max(MARKERS_X, MARKERS_Y) * (MARKER_LENGTH_M + MARKER_SEPARATION_M)
                axis_len = board_size_max * 0.6
                draw_axes(img, K, rvec_board, tvec_board,
                          axis_len=axis_len, thickness=3)
                # ---- box 中心座標の算出 (board 左上原点 -> 中心に移動 -> X軸回転) ----
                R_flip_x = np.array([[1, 0, 0],
                                     [0, -1, 0],
                                     [0, 0, -1]], dtype=np.float32)  # X周り180度
                center_offset = np.array([BOARD_W_M * 0.5,
                                          BOARD_H_M * 0.5,
                                          0.0], dtype=np.float32)
                z_offset = np.array([0.0, 0.0, -BOX_DEPTH_M * 0.5], dtype=np.float32)
                # 新フレーム (box) -> board
                t_board_box = center_offset + R_flip_x @ z_offset
                R_cam_box = R_board @ R_flip_x
                t_cam_box = R_board @ t_board_box.reshape(3, 1) + tvec_board.reshape(3, 1)
                rvec_box, _ = cv2.Rodrigues(R_cam_box)
                tvec_box = t_cam_box.astype(np.float32)
                box_center_cam = t_cam_box.flatten()
                box_pose_ok = True
                if time.time() - last_print > 1.0:
                    print("cam->box pose:")
                    print(f"box center (cam): {box_center_cam}")
                    print(f"box rpy (cam deg): {rotmat_to_rpy_deg(R_cam_box)}")
                    last_print = time.time()
                # 画像上に box の座標軸と中心を可視化
                draw_axes(img, K, rvec_box, tvec_box,
                          axis_len=board_size_max * 0.5, thickness=3)
                proj_center, _ = cv2.projectPoints(
                    np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
                    rvec_box, tvec_box, K, None
                )
                cx, cy = proj_center[0, 0].astype(int)
                cv2.circle(img, (cx, cy), 8, (0, 200, 255), -1)
        # ==== 左側パネル（情報表示） ==========================================
        panel_w = 420
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (panel_w, RES_H), (0, 0, 0), -1)
        img = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)
        # 見出し
        cv2.putText(img, f"D435 {RES_W}x{RES_H}@{FPS}  FPS: {fps:.1f}",
                    (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, f"Serial: {USE_SERIAL}",
                    (12, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (220, 220, 220), 2, cv2.LINE_AA)
        y = 86
        cv2.putText(img,
                    f"ArUco GridBoard {MARKERS_X}x{MARKERS_Y}  len={MARKER_LENGTH_M*1000:.2f}mm",
                    (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.60,
                    (255, 255, 255), 2, cv2.LINE_AA)
        y += 24
        cv2.putText(img,
                    f"Detected markers: {len(used_ids)}",
                    (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.60,
                    (230, 230, 230), 2, cv2.LINE_AA)
        y += 24
        if board_pose_ok:
            line1 = f"Board dist = {dist_board:.3f} m"
            line2 = f"RPY = ({rpy_board[0]:.1f}, {rpy_board[1]:.1f}, {rpy_board[2]:.1f}) deg"
            cv2.putText(img, line1, (12, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.60,
                        (255, 255, 255), 2, cv2.LINE_AA)
            y += 24
            cv2.putText(img, line2, (12, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.60,
                        (230, 230, 230), 2, cv2.LINE_AA)
            y += 24
            if box_pose_ok and box_center_cam is not None:
                line3 = (f"Box center (cam): "
                         f"({box_center_cam[0]:.3f}, {box_center_cam[1]:.3f}, {box_center_cam[2]:.3f}) m")
                cv2.putText(img, line3, (12, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (180, 255, 180), 2, cv2.LINE_AA)
                y += 24
        else:
            cv2.putText(img, "Board pose: ---",
                        (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.60,
                        (200, 200, 200), 2, cv2.LINE_AA)
            y += 24
        # ---- FPS 更新 ----
        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(1e-6, now - t_prev))
        t_prev = now
        cv2.imshow("ArUco GridBoard (RealSense D435)", img)
        k = cv2.waitKey(1) & 0xFF
        if k in (27, ord('q')):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
