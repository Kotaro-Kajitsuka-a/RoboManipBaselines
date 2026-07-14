import numpy as np
from scipy.spatial.transform import Rotation as R


def load_calib(path):
    """
    calib ファイルを読み込み、4×4 の同次変換行列を返す。
    3×4 しか書かれていない形式に対応（最後の行は [0 0 0 1] を補う）
    """
    data = np.loadtxt(path)
    if data.shape == (3, 4):
        T = np.eye(4)
        T[:3, :4] = data
    elif data.shape == (4, 4):
        T = data
    else:
        raise ValueError(f"Unexpected shape in {path}: {data.shape}")
    return T


def average_rotation(R_L, R_R):
    """SO(3) のクォータニオン平均（向きを合わせてから平均）"""
    q_L = R.from_matrix(R_L).as_quat()  # [x y z w]
    q_R = R.from_matrix(R_R).as_quat()

    # 内積が負なら逆向き → 片方に - を付けて同じ半球に合わせる
    if np.dot(q_L, q_R) < 0:
        q_R = -q_R

    # 2つのクォータニオンを単純平均し、正規化
    q_C = q_L + q_R
    q_C /= np.linalg.norm(q_C)

    return R.from_quat(q_C).as_matrix()


def compute_center_transform(left_path, right_path):
    # 変換行列を読み込み
    T_L = load_calib(left_path)
    T_R = load_calib(right_path)

    R_L, p_L = T_L[:3, :3], T_L[:3, 3]
    R_R, p_R = T_R[:3, :3], T_R[:3, 3]

    # 位置の平均（中点）
    p_C = 0.5 * (p_L + p_R)

    # 回転の平均（クォータニオン法）
    R_C = average_rotation(R_L, R_R)

    # 4×4 の同次変換行列を構築
    T_C = np.eye(4)
    T_C[:3, :3] = R_C
    T_C[:3, 3] = p_C

    return T_C


if __name__ == "__main__":
    # 例
    left_file = "robo_manip_baselines/calib/base_left_T_neo_front.calib"
    right_file = "robo_manip_baselines/calib/base_right_T_neo_front.calib"

    T_center = compute_center_transform(left_file, right_file)
    print("T_cam_to_center:")
    print(T_center)

    # 保存（入力形式に合わせて 3×4 で保存）
    output_path = "robo_manip_baselines/calib/base_center_T_neo_front.calib"
    np.savetxt(output_path, T_center[:4, :4], fmt="%.8f")

    print(f"\nSaved center transform to: {output_path}")
