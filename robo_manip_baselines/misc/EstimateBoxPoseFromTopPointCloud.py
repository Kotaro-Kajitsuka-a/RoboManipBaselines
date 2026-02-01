import argparse

import numpy as np
import open3d as o3d
import pinocchio as pin

from robo_manip_baselines.common import DataKey, RmbData
from robo_manip_baselines.common.utils.MathUtils import get_pose_from_rot_pos


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("rmb_path", type=str, help="path to .rmb or .hdf5 file")
    parser.add_argument("--camera_name", type=str, default="top")
    parser.add_argument("--time_idx", type=int, default=0)
    parser.add_argument("--voxel_size", type=float, default=0.0)
    parser.add_argument("--min_depth", type=float, default=None)
    parser.add_argument("--max_depth", type=float, default=None)
    parser.add_argument("--remove_plane", action="store_true")
    parser.add_argument("--plane_distance", type=float, default=0.005)
    parser.add_argument("--plane_ransac_n", type=int, default=3)
    parser.add_argument("--plane_iterations", type=int, default=200)
    parser.add_argument("--use_dbscan", action="store_true")
    parser.add_argument("--dbscan_eps", type=float, default=0.01)
    parser.add_argument("--dbscan_min_points", type=int, default=30)
    parser.add_argument("--outlier_nb_neighbors", type=int, default=20)
    parser.add_argument("--outlier_std_ratio", type=float, default=2.0)
    parser.add_argument("--robust_obb", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    return parser.parse_args()


def load_pointcloud(rmb_path, camera_name, time_idx):
    pc_key = DataKey.get_pointcloud_key(camera_name)
    with RmbData(rmb_path) as rmb_data:
        if pc_key not in rmb_data.keys():
            raise KeyError(f"Pointcloud key not found: {pc_key}")
        pointcloud_seq = rmb_data[pc_key][:]

    if time_idx < 0 or time_idx >= len(pointcloud_seq):
        raise IndexError(f"time_idx out of range: {time_idx}")
    return pointcloud_seq[time_idx]


def to_open3d_pointcloud(pointcloud):
    if pointcloud.shape[1] < 3:
        raise ValueError(f"Invalid pointcloud shape: {pointcloud.shape}")
    xyz = pointcloud[:, :3].astype(np.float64)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))

    if pointcloud.shape[1] >= 6:
        colors = pointcloud[:, 3:6].astype(np.float64)
        if colors.max() > 1.0:
            colors = colors / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def preprocess_pointcloud(pcd, args):
    cleaned = pcd.remove_non_finite_points()
    if isinstance(cleaned, tuple):
        pcd = cleaned[0]
    else:
        pcd = cleaned

    if args.voxel_size > 0:
        pcd = pcd.voxel_down_sample(args.voxel_size)

    if args.min_depth is not None or args.max_depth is not None:
        points = np.asarray(pcd.points)
        min_depth = -np.inf if args.min_depth is None else args.min_depth
        max_depth = np.inf if args.max_depth is None else args.max_depth
        depth_mask = (min_depth <= points[:, 2]) & (points[:, 2] <= max_depth)
        if not np.any(depth_mask):
            raise RuntimeError("Depth filter removed all points.")
        pcd = pcd.select_by_index(np.where(depth_mask)[0])

    if args.outlier_nb_neighbors > 0 and args.outlier_std_ratio > 0:
        pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors=args.outlier_nb_neighbors,
            std_ratio=args.outlier_std_ratio,
        )

    if args.remove_plane:
        _, inliers = pcd.segment_plane(
            distance_threshold=args.plane_distance,
            ransac_n=args.plane_ransac_n,
            num_iterations=args.plane_iterations,
        )
        pcd = pcd.select_by_index(inliers, invert=True)

    if args.use_dbscan:
        labels = np.array(
            pcd.cluster_dbscan(
                eps=args.dbscan_eps, min_points=args.dbscan_min_points
            )
        )
        valid_mask = labels >= 0
        if not np.any(valid_mask):
            raise RuntimeError("DBSCAN found no clusters.")
        unique_labels, counts = np.unique(labels[valid_mask], return_counts=True)
        largest_label = unique_labels[np.argmax(counts)]
        pcd = pcd.select_by_index(np.where(labels == largest_label)[0])

    if len(pcd.points) == 0:
        raise RuntimeError("Pointcloud is empty after preprocessing.")

    return pcd


def estimate_box_pose(pcd, robust):
    obb = pcd.get_oriented_bounding_box(robust=robust)
    center = np.asarray(obb.center)
    rotation = np.asarray(obb.R)
    extent = np.asarray(obb.extent)
    return obb, center, rotation, extent


def print_pose(center, rotation, extent):
    pose = get_pose_from_rot_pos(rotation, center)
    rpy = pin.rpy.matrixToRpy(rotation)
    rpy_deg = np.rad2deg(rpy)

    np.set_printoptions(precision=6, suppress=True)
    print("[Result] center (x, y, z):", center)
    print("[Result] extent (dx, dy, dz):", extent)
    print("[Result] rotation matrix:\n", rotation)
    print("[Result] pose (tx, ty, tz, qw, qx, qy, qz):", pose)
    print("[Result] rpy (rad):", rpy)
    print("[Result] rpy (deg):", rpy_deg)


def visualize(pcd, obb):
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    obb.color = (1.0, 0.0, 0.0)
    o3d.visualization.draw_geometries([pcd, obb, coord])


def main():
    args = parse_args()
    pointcloud = load_pointcloud(args.rmb_path, args.camera_name, args.time_idx)
    pcd = to_open3d_pointcloud(pointcloud)
    pcd = preprocess_pointcloud(pcd, args)
    obb, center, rotation, extent = estimate_box_pose(pcd, args.robust_obb)
    print_pose(center, rotation, extent)

    if args.visualize:
        visualize(pcd, obb)


if __name__ == "__main__":
    main()
