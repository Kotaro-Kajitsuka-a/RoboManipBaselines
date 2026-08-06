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
from robo_manip_baselines.policy.wrench_predictor4_online.WrenchPredictor4OnlineUtils import (
    NUM_LIFTING_OBJECTS,
    load_model_meta_info,
    load_pb,
    load_policy,
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
        help=(
            f"replace an existing {DataKey.MATERIAL_PROPERTY} dataset if its "
            "value differs"
        ),
    )
    return parser.parse_args()


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

    with RmbData(filename) as rmb_data:
        num_steps = rmb_data[DataKey.TIME].shape[0]

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
    rmb_path: str,
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
    with RmbData(rmb_path, mode="r+") as rmb_data:
        h5file = rmb_data.h5file
        assert pb_trajectory.shape[0] == h5file[DataKey.TIME].shape[0], (
            pb_trajectory.shape,
            h5file[DataKey.TIME].shape,
        )

        if DataKey.MATERIAL_PROPERTY in h5file:
            existing = h5file[DataKey.MATERIAL_PROPERTY][:]
            if existing.shape == pb_trajectory.shape and np.array_equal(
                existing,
                pb_trajectory,
            ):
                status = "unchanged"
            else:
                assert overwrite, (
                    f"{rmb_path}: {DataKey.MATERIAL_PROPERTY} already exists "
                    "with a different value; pass --overwrite to replace it"
                )
                del h5file[DataKey.MATERIAL_PROPERTY]
                h5file.create_dataset(DataKey.MATERIAL_PROPERTY, data=pb_trajectory)
                status = "overwritten"
        else:
            h5file.create_dataset(DataKey.MATERIAL_PROPERTY, data=pb_trajectory)
            status = "added"

        dataset = h5file[DataKey.MATERIAL_PROPERTY]
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
        model_meta_info,
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
        status = write_online_pb(
            filename,
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
            f"[{episode_idx}/{len(filenames)}] [{status}] {filename} | "
            f"updates={num_updates}, PB={initial_pb.tolist()} -> {final_pb:.6f}"
        )

    print(f"device: {device}")
    print(
        f"initial object: {initial_object_key} "
        f"(id={args.initial_object_id}, PB={initial_pb.tolist()})"
    )
    print(f"HDF5 key: {DataKey.MATERIAL_PROPERTY}")
    print(
        "episodes: "
        f"{len(filenames)} "
        f"(added={counts['added']}, overwritten={counts['overwritten']}, "
        f"unchanged={counts['unchanged']})"
    )


if __name__ == "__main__":
    main()
