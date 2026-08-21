import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from robo_manip_baselines.common import (
    DataKey,
    RmbData,
    find_rmb_files,
    set_random_seed,
)
from robo_manip_baselines.policy.wrench_predictor4.WrenchPredictor4Model import (
    WrenchPredictor4Model,
)
from robo_manip_baselines.policy.wrench_predictor4_online.WrenchPredictor4OnlineDataset import (
    WrenchPredictor4OnlineDataset,
)
from robo_manip_baselines.policy.wrench_predictor4_online.WrenchPredictor4OnlineUtils import (
    ONLINE_PB_STD_KEY,
    GaussianBeliefOnlinePb,
    calculate_pb_candidate_losses,
    load_model_meta_info,
    load_pb,
    load_policy,
    resolve_gaussian_num_points,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate PB online in each RMB episode and add its causal trajectory "
            "to the HDF5 data."
        )
    )
    parser.add_argument(
        "dataset_path",
        type=Path,
        help="RMB episode, HDF5 file, or directory containing RMB episodes",
    )
    parser.add_argument(
        "initial_object_id",
        type=int,
        help=(
            "trained PB ID used before and at the start of online adaptation; "
            "valid IDs are read from the checkpoint"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="pretrained WP4 checkpoint used for online PB adaptation",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-2,
        help="learning rate applied only to the online PB",
    )
    parser.add_argument(
        "--update_type",
        choices=("adam", "gaussian_belief"),
        default="adam",
        help="online PB update rule",
    )
    parser.add_argument(
        "--initial_std",
        type=float,
        default=None,
        help="initial PB standard deviation; required for gaussian_belief",
    )
    parser.add_argument(
        "--num_points",
        type=int,
        default=None,
        help="Gauss-Hermite order (default: 16 times the PB dimension)",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="pseudo-likelihood inverse temperature for gaussian_belief",
    )
    parser.add_argument(
        "--wrench_loss_weight",
        type=float,
        default=0.0,
        help="weight of the normalized wrench prediction loss used to adapt PB",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=f"replace an existing {DataKey.MATERIAL_PROPERTY} dataset",
    )
    return parser.parse_args()


def adapt_pb_trajectory(
    filename: str,
    initial_pb: np.ndarray,
    policy: WrenchPredictor4Model,
    model_meta_info: dict,
    device: torch.device,
    learning_rate: float,
    wrench_loss_weight: float,
    update_type: str = "adam",
    initial_std: float | None = None,
    num_points: int | None = None,
    beta: float = 1.0,
) -> tuple[np.ndarray, np.ndarray | None, int, np.ndarray, np.ndarray | None]:
    dataset = WrenchPredictor4OnlineDataset(
        [filename],
        model_meta_info,
        enable_rmb_cache=False,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    with RmbData(filename) as rmb_data:
        num_steps = rmb_data[DataKey.TIME].shape[0]

    pb_trajectory = np.broadcast_to(
        initial_pb,
        (num_steps, initial_pb.shape[0]),
    ).copy()
    pb_std_trajectory = None
    if update_type == "adam":
        online_pb = torch.nn.Parameter(
            torch.tensor(initial_pb, dtype=torch.float32, device=device)
        )
        optimizer = torch.optim.Adam([online_pb], lr=learning_rate)
        gaussian_belief = None
    else:
        assert update_type == "gaussian_belief", update_type
        assert initial_std is not None, initial_std
        num_points = resolve_gaussian_num_points(initial_pb.shape[0], num_points)
        gaussian_belief = GaussianBeliefOnlinePb(
            initial_pb,
            initial_std,
            num_points,
            beta,
            device,
        )
        online_pb = gaussian_belief.mean
        optimizer = None
        pb_std_trajectory = np.full_like(pb_trajectory, initial_std)

    skip = model_meta_info["data"]["skip"]
    horizon = model_meta_info["data"]["horizon"]
    num_updates = 0
    for window_idx, batch in enumerate(dataloader):
        batch = {key: value.to(device) for key, value in batch.items()}

        start_time_idx = dataset.start_time_idxes[window_idx]
        if update_type == "adam":
            optimizer.zero_grad()
            prediction = policy(batch, online_pb.unsqueeze(0))
            start = policy.n_obs_steps
            # Use the same weighted pose and wrench prediction objective as WP4 training.
            pose_loss = F.mse_loss(
                prediction["image_feature"][:, start:],
                batch["image_feature"][:, start:],
            )
            wrench_loss = F.mse_loss(
                prediction["wrench"][:, start:],
                batch["wrench"][:, start:],
            )
            loss = pose_loss + wrench_loss_weight * wrench_loss
            loss.backward()
            assert online_pb.grad is not None
            optimizer.step()
            num_updates += 1
        else:
            with torch.no_grad():
                losses = calculate_pb_candidate_losses(
                    policy,
                    batch,
                    gaussian_belief.get_candidates(),
                    wrench_loss_weight,
                )
                gaussian_belief.update(losses)
            num_updates += 1

        end_raw_idx = (start_time_idx + horizon - 1) * skip
        assert end_raw_idx < num_steps, (end_raw_idx, num_steps)

        # The updated PB becomes available when the complete prediction window
        # has been observed. Hold it forward until the next skipped observation;
        # never copy it into frames preceding the window endpoint.
        next_raw_idx = min(end_raw_idx + skip, num_steps)
        pb_trajectory[end_raw_idx:next_raw_idx] = online_pb.detach().cpu().numpy()
        if pb_std_trajectory is not None:
            pb_std_trajectory[end_raw_idx:next_raw_idx] = (
                gaussian_belief.std.detach().cpu().numpy()
            )

    final_std = None
    if gaussian_belief is not None:
        final_std = gaussian_belief.std.detach().cpu().numpy().copy()
    return (
        pb_trajectory,
        pb_std_trajectory,
        num_updates,
        online_pb.detach().cpu().numpy().copy(),
        final_std,
    )


def write_online_pb(
    rmb_path: str,
    pb_trajectory: np.ndarray,
    pb_std_trajectory: np.ndarray | None,
    initial_pb: np.ndarray,
    initial_object_id: int,
    initial_object_key: str,
    checkpoint_path: Path,
    model_meta_info: dict,
    learning_rate: float,
    wrench_loss_weight: float,
    num_updates: int,
    final_pb: np.ndarray,
    final_pb_std: np.ndarray | None,
    update_type: str,
    initial_std: float | None,
    num_points: int,
    beta: float,
    overwrite: bool,
) -> str:
    pb_dim = pb_trajectory.shape[1]
    with RmbData(rmb_path, mode="r+") as rmb_data:
        h5file = rmb_data.h5file
        assert pb_trajectory.shape[0] == h5file[DataKey.TIME].shape[0], (
            pb_trajectory.shape,
            h5file[DataKey.TIME].shape,
        )

        if DataKey.MATERIAL_PROPERTY in h5file:
            if not overwrite:
                raise FileExistsError(
                    f"{rmb_path}: {DataKey.MATERIAL_PROPERTY} already exists "
                    "and --overwrite was not specified"
                )
            del h5file[DataKey.MATERIAL_PROPERTY]
            if ONLINE_PB_STD_KEY in h5file:
                del h5file[ONLINE_PB_STD_KEY]
            status = "overwritten"
        else:
            status = "added"
        h5file.create_dataset(DataKey.MATERIAL_PROPERTY, data=pb_trajectory)
        if pb_std_trajectory is not None:
            h5file.create_dataset(ONLINE_PB_STD_KEY, data=pb_std_trajectory)

        dataset = h5file[DataKey.MATERIAL_PROPERTY]
        dataset.attrs["pb_dim"] = pb_dim
        dataset.attrs["initial_object_id"] = initial_object_id
        dataset.attrs["initial_object_key"] = initial_object_key
        dataset.attrs["initial_pb"] = initial_pb
        dataset.attrs["final_pb"] = final_pb
        dataset.attrs["source_checkpoint"] = str(checkpoint_path.resolve())
        dataset.attrs["online_update_type"] = update_type
        if update_type == "adam":
            dataset.attrs["online_learning_rate"] = learning_rate
        dataset.attrs["online_wrench_loss_weight"] = wrench_loss_weight
        dataset.attrs["online_num_updates"] = num_updates
        dataset.attrs["online_skip"] = model_meta_info["data"]["skip"]
        dataset.attrs["online_horizon"] = model_meta_info["data"]["horizon"]
        dataset.attrs["online_n_obs_steps"] = model_meta_info["data"]["n_obs_steps"]
        dataset.attrs["online_update_alignment"] = "window_end_forward_hold"
        if pb_std_trajectory is not None:
            std_dataset = h5file[ONLINE_PB_STD_KEY]
            std_dataset.attrs["initial_std"] = initial_std
            std_dataset.attrs["final_std"] = final_pb_std
            std_dataset.attrs["gauss_hermite_num_points"] = num_points
            std_dataset.attrs["pseudo_likelihood_beta"] = beta
            std_dataset.attrs["update_interval_windows"] = 1
            std_dataset.attrs["overlapping_evidence"] = True

    return status


def main() -> None:
    args = parse_args()
    if args.update_type == "adam":
        assert args.lr > 0.0, args.lr
    else:
        assert args.initial_std is not None and args.initial_std > 0.0, args.initial_std
        if args.num_points is not None:
            assert args.num_points >= 3, args.num_points
        assert args.beta > 0.0, args.beta
    set_random_seed(args.seed)

    filenames = find_rmb_files(str(args.dataset_path))
    assert len(filenames) > 0, args.dataset_path
    checkpoint_path = args.checkpoint.resolve()
    model_meta_info = load_model_meta_info(checkpoint_path)
    initial_pb, initial_object_key = load_pb(
        checkpoint_path,
        args.initial_object_id,
        model_meta_info,
    )
    pb_dim = model_meta_info["policy"]["args"]["pb_dim"]
    assert initial_pb.shape == (pb_dim,), (initial_pb.shape, pb_dim)
    args.num_points = resolve_gaussian_num_points(pb_dim, args.num_points)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = load_policy(checkpoint_path, model_meta_info, device)
    counts = {"added": 0, "overwritten": 0}

    for episode_idx, filename in enumerate(filenames, start=1):
        (
            pb_trajectory,
            pb_std_trajectory,
            num_updates,
            final_pb,
            final_pb_std,
        ) = adapt_pb_trajectory(
            filename,
            initial_pb,
            policy,
            model_meta_info,
            device,
            args.lr,
            args.wrench_loss_weight,
            args.update_type,
            args.initial_std,
            args.num_points,
            args.beta,
        )
        status = write_online_pb(
            filename,
            pb_trajectory,
            pb_std_trajectory,
            initial_pb,
            args.initial_object_id,
            initial_object_key,
            checkpoint_path,
            model_meta_info,
            args.lr,
            args.wrench_loss_weight,
            num_updates,
            final_pb,
            final_pb_std,
            args.update_type,
            args.initial_std,
            args.num_points,
            args.beta,
            args.overwrite,
        )
        counts[status] += 1
        print(
            f"[{episode_idx}/{len(filenames)}] [{status}] {filename} | "
            f"updates={num_updates}, PB={initial_pb.tolist()} -> {final_pb.tolist()}, "
            f"std={None if final_pb_std is None else final_pb_std.tolist()}"
        )

    print(f"device: {device}")
    print(
        f"initial object: {initial_object_key} "
        f"(id={args.initial_object_id}, PB={initial_pb.tolist()})"
    )
    print(f"HDF5 key: {DataKey.MATERIAL_PROPERTY}")
    print(f"online update type: {args.update_type}")
    print(f"online wrench loss weight: {args.wrench_loss_weight}")
    print(
        "episodes: "
        f"{len(filenames)} "
        f"(added={counts['added']}, overwritten={counts['overwritten']})"
    )


if __name__ == "__main__":
    main()
