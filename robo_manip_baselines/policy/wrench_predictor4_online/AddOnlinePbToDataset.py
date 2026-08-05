import argparse
import pickle
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from robo_manip_baselines.common import find_rmb_files, set_random_seed
from robo_manip_baselines.policy.wrench_predictor4.WrenchPredictor4Model import (
    WrenchPredictor4Model,
)
from robo_manip_baselines.policy.wrench_predictor4_online.AddConstantPbToDataset import (
    DATA_KEY,
    NUM_LIFTING_OBJECTS,
    get_hdf5_path,
    load_pb,
)
from robo_manip_baselines.policy.wrench_predictor4_online.WrenchPredictor4OnlineDataset import (
    WrenchPredictor4OnlineDataset,
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
        choices=range(NUM_LIFTING_OBJECTS),
        help="trained PB ID used before and at the start of online adaptation",
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
        "--seed",
        type=int,
        default=42,
        help="random seed",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=f"replace an existing {DATA_KEY} dataset if its value differs",
    )
    return parser.parse_args()


def load_model_meta_info(checkpoint_path: Path) -> dict:
    checkpoint_path = checkpoint_path.resolve()
    assert checkpoint_path.is_file(), checkpoint_path

    meta_info_path = checkpoint_path.parent / "model_meta_info.pkl"
    assert meta_info_path.is_file(), meta_info_path
    with meta_info_path.open("rb") as file:
        model_meta_info = pickle.load(file)

    assert model_meta_info["policy"]["name"] == "WrenchPredictor4"
    policy_args = model_meta_info["policy"]["args"]
    assert policy_args["pb_dim"] == 1, policy_args["pb_dim"]
    assert policy_args["num_objects"] >= NUM_LIFTING_OBJECTS, policy_args
    assert model_meta_info["data"]["horizon"] > 0
    assert model_meta_info["data"]["skip"] > 0
    return model_meta_info


def load_policy(
    checkpoint_path: Path,
    model_meta_info: dict,
    device: torch.device,
) -> WrenchPredictor4Model:
    policy = WrenchPredictor4Model.from_policy_args(
        model_meta_info["policy"]["args"]
    ).to(device)
    policy.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )
    )
    policy.requires_grad_(False)
    policy.eval()
    return policy


def adapt_pb_trajectory(
    filename: str,
    initial_pb: np.ndarray,
    policy: WrenchPredictor4Model,
    model_meta_info: dict,
    device: torch.device,
    learning_rate: float,
) -> tuple[np.ndarray, int, float]:
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

    hdf5_path = get_hdf5_path(filename)
    with h5py.File(hdf5_path, "r") as h5file:
        assert "time" in h5file, hdf5_path
        num_steps = h5file["time"].shape[0]

    pb_trajectory = np.broadcast_to(
        initial_pb,
        (num_steps, initial_pb.shape[0]),
    ).copy()
    online_pb = torch.nn.Parameter(
        torch.tensor(initial_pb, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.Adam([online_pb], lr=learning_rate)

    skip = model_meta_info["data"]["skip"]
    horizon = model_meta_info["data"]["horizon"]
    for window_idx, batch in enumerate(dataloader):
        batch = {key: value.to(device) for key, value in batch.items()}

        optimizer.zero_grad()
        prediction = policy(batch, online_pb.unsqueeze(0))
        start = policy.n_obs_steps
        # Online PB is identified only from the future object-pose prediction
        # error. Wrench prediction is intentionally excluded from this loss.
        pose_loss = F.mse_loss(
            prediction["image_feature"][:, start:],
            batch["image_feature"][:, start:],
        )
        pose_loss.backward()
        assert online_pb.grad is not None
        optimizer.step()

        start_time_idx = dataset.start_time_idxes[window_idx]
        end_raw_idx = (start_time_idx + horizon - 1) * skip
        assert end_raw_idx < num_steps, (end_raw_idx, num_steps)

        # The updated PB becomes available when the complete prediction window
        # has been observed. Hold it forward until the next skipped observation;
        # never copy it into frames preceding the window endpoint.
        next_raw_idx = min(end_raw_idx + skip, num_steps)
        pb_trajectory[end_raw_idx:next_raw_idx] = online_pb.detach().cpu().numpy()

    return pb_trajectory, len(dataset), online_pb.detach().item()


def write_online_pb(
    hdf5_path: Path,
    pb_trajectory: np.ndarray,
    initial_pb: np.ndarray,
    initial_object_id: int,
    initial_object_key: str,
    checkpoint_path: Path,
    model_meta_info: dict,
    learning_rate: float,
    num_updates: int,
    final_pb: float,
    overwrite: bool,
) -> str:
    assert hdf5_path.is_file(), hdf5_path
    with h5py.File(hdf5_path, "r+") as h5file:
        assert "time" in h5file, hdf5_path
        assert pb_trajectory.shape[0] == h5file["time"].shape[0], (
            pb_trajectory.shape,
            h5file["time"].shape,
        )

        if DATA_KEY in h5file:
            existing = h5file[DATA_KEY][:]
            if existing.shape == pb_trajectory.shape and np.array_equal(
                existing,
                pb_trajectory,
            ):
                status = "unchanged"
            else:
                assert overwrite, (
                    f"{hdf5_path}: {DATA_KEY} already exists with a different "
                    "value; pass --overwrite to replace it"
                )
                del h5file[DATA_KEY]
                h5file.create_dataset(DATA_KEY, data=pb_trajectory)
                status = "overwritten"
        else:
            h5file.create_dataset(DATA_KEY, data=pb_trajectory)
            status = "added"

        dataset = h5file[DATA_KEY]
        dataset.attrs["initial_object_id"] = initial_object_id
        dataset.attrs["initial_object_key"] = initial_object_key
        dataset.attrs["initial_pb"] = initial_pb
        dataset.attrs["final_pb"] = final_pb
        dataset.attrs["source_checkpoint"] = str(checkpoint_path.resolve())
        dataset.attrs["online_learning_rate"] = learning_rate
        dataset.attrs["online_num_updates"] = num_updates
        dataset.attrs["online_skip"] = model_meta_info["data"]["skip"]
        dataset.attrs["online_horizon"] = model_meta_info["data"]["horizon"]
        dataset.attrs["online_n_obs_steps"] = model_meta_info["data"]["n_obs_steps"]
        dataset.attrs["online_update_alignment"] = "window_end_forward_hold"

    return status


def main() -> None:
    args = parse_args()
    assert args.lr > 0.0, args.lr
    set_random_seed(args.seed)

    filenames = find_rmb_files(str(args.dataset_path))
    assert len(filenames) > 0, args.dataset_path
    checkpoint_path = args.checkpoint.resolve()
    model_meta_info = load_model_meta_info(checkpoint_path)
    initial_pb, initial_object_key = load_pb(
        checkpoint_path,
        args.initial_object_id,
    )
    pb_dim = model_meta_info["policy"]["args"]["pb_dim"]
    assert initial_pb.shape == (pb_dim,), (initial_pb.shape, pb_dim)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = load_policy(checkpoint_path, model_meta_info, device)
    counts = {"added": 0, "overwritten": 0, "unchanged": 0}

    for episode_idx, filename in enumerate(filenames, start=1):
        pb_trajectory, num_updates, final_pb = adapt_pb_trajectory(
            filename,
            initial_pb,
            policy,
            model_meta_info,
            device,
            args.lr,
        )
        hdf5_path = get_hdf5_path(filename)
        status = write_online_pb(
            hdf5_path,
            pb_trajectory,
            initial_pb,
            args.initial_object_id,
            initial_object_key,
            checkpoint_path,
            model_meta_info,
            args.lr,
            num_updates,
            final_pb,
            args.overwrite,
        )
        counts[status] += 1
        print(
            f"[{episode_idx}/{len(filenames)}] [{status}] {hdf5_path} | "
            f"updates={num_updates}, PB={initial_pb.tolist()} -> {final_pb:.6f}"
        )

    print(f"device: {device}")
    print(
        f"initial object: {initial_object_key} "
        f"(id={args.initial_object_id}, PB={initial_pb.tolist()})"
    )
    print(f"HDF5 key: {DATA_KEY}")
    print(
        "episodes: "
        f"{len(filenames)} "
        f"(added={counts['added']}, overwritten={counts['overwritten']}, "
        f"unchanged={counts['unchanged']})"
    )


if __name__ == "__main__":
    main()
