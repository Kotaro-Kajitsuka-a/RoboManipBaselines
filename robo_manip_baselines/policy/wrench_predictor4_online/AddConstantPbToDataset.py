import argparse
import pickle
from pathlib import Path

import h5py
import numpy as np
import torch

from robo_manip_baselines.common import find_rmb_files


DATA_KEY = "material_property"
NUM_LIFTING_OBJECTS = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Add one trained, constant PB to every timestep of each RMB episode."
        )
    )
    parser.add_argument(
        "dataset_path",
        type=Path,
        help="RMB episode, HDF5 file, or directory containing RMB episodes",
    )
    parser.add_argument(
        "object_id",
        type=int,
        choices=range(NUM_LIFTING_OBJECTS),
        help="trained PB ID: 0, 1, or 2",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="WP4 checkpoint containing material_property.weight",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=f"replace an existing {DATA_KEY} dataset if its value differs",
    )
    return parser.parse_args()


def load_pb(checkpoint_path: Path, object_id: int) -> tuple[np.ndarray, str]:
    checkpoint_path = checkpoint_path.resolve()
    assert checkpoint_path.is_file(), checkpoint_path

    meta_info_path = checkpoint_path.parent / "model_meta_info.pkl"
    assert meta_info_path.is_file(), meta_info_path
    with meta_info_path.open("rb") as file:
        meta_info = pickle.load(file)

    object_key = f"WrenchPredObject{object_id}"
    object_key_to_id = meta_info["material_property"]["object_key_to_id"]
    assert object_key_to_id[object_key] == object_id, (
        object_key,
        object_key_to_id,
    )

    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    material_property = state_dict["material_property.weight"]
    pb_dim = meta_info["material_property"]["pb_dim"]
    assert material_property.ndim == 2, material_property.shape
    assert material_property.shape[1] == pb_dim, (
        material_property.shape,
        pb_dim,
    )
    assert object_id < material_property.shape[0], (
        object_id,
        material_property.shape,
    )

    pb = material_property[object_id].detach().numpy().astype(np.float32)
    return pb, object_key


def get_hdf5_path(rmb_path: str) -> Path:
    path = Path(rmb_path)
    if path.suffix == ".hdf5":
        return path
    assert path.suffix == ".rmb", path
    return path / "main.rmb.hdf5"


def write_constant_pb(
    hdf5_path: Path,
    pb: np.ndarray,
    object_id: int,
    object_key: str,
    checkpoint_path: Path,
    overwrite: bool,
) -> str:
    assert hdf5_path.is_file(), hdf5_path
    with h5py.File(hdf5_path, "r+") as h5file:
        assert "time" in h5file, hdf5_path
        num_steps = h5file["time"].shape[0]
        data = np.broadcast_to(pb, (num_steps, pb.shape[0])).copy()

        if DATA_KEY in h5file:
            existing = h5file[DATA_KEY][:]
            if existing.shape == data.shape and np.array_equal(existing, data):
                status = "unchanged"
            else:
                assert overwrite, (
                    f"{hdf5_path}: {DATA_KEY} already exists with a different "
                    "value; pass --overwrite to replace it"
                )
                del h5file[DATA_KEY]
                h5file.create_dataset(DATA_KEY, data=data)
                status = "overwritten"
        else:
            h5file.create_dataset(DATA_KEY, data=data)
            status = "added"

        dataset = h5file[DATA_KEY]
        dataset.attrs["object_id"] = object_id
        dataset.attrs["object_key"] = object_key
        dataset.attrs["source_checkpoint"] = str(checkpoint_path.resolve())

    return status


def main() -> None:
    args = parse_args()
    filenames = find_rmb_files(str(args.dataset_path))
    assert len(filenames) > 0, args.dataset_path

    pb, object_key = load_pb(args.checkpoint, args.object_id)
    counts = {"added": 0, "overwritten": 0, "unchanged": 0}
    for filename in filenames:
        hdf5_path = get_hdf5_path(filename)
        status = write_constant_pb(
            hdf5_path,
            pb,
            args.object_id,
            object_key,
            args.checkpoint,
            args.overwrite,
        )
        counts[status] += 1
        print(f"[{status}] {hdf5_path}")

    print(f"object: {object_key} (id={args.object_id})")
    print(f"PB: {pb.tolist()}")
    print(f"HDF5 key: {DATA_KEY}, shape per episode: (T, {pb.shape[0]})")
    print(
        "episodes: "
        f"{len(filenames)} "
        f"(added={counts['added']}, overwritten={counts['overwritten']}, "
        f"unchanged={counts['unchanged']})"
    )


if __name__ == "__main__":
    main()
