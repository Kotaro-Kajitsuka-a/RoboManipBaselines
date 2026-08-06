import argparse
from pathlib import Path

import numpy as np

from robo_manip_baselines.common import DataKey, RmbData, find_rmb_files
from robo_manip_baselines.policy.wrench_predictor4.WrenchPredictor4Dataset import (
    WrenchPredictor4Dataset,
)
from robo_manip_baselines.policy.wrench_predictor4_online.WrenchPredictor4OnlineUtils import (
    get_wp4_provenance,
    load_pb_table,
    write_wp4_provenance_attrs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Add each object's trained, constant PB to every timestep of its "
            "RMB episodes."
        )
    )
    parser.add_argument(
        "dataset_path",
        type=Path,
        help="RMB episode, HDF5 file, or directory containing RMB episodes",
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
        help=(
            f"replace an existing {DataKey.MATERIAL_PROPERTY} dataset if its "
            "value differs"
        ),
    )
    return parser.parse_args()


def write_constant_pb(
    rmb_path: str,
    pb: np.ndarray,
    object_id: int,
    object_key: str,
    wp4_provenance: dict[str, str],
    overwrite: bool,
) -> str:
    with RmbData(rmb_path, mode="r+") as rmb_data:
        h5file = rmb_data.h5file
        assert DataKey.TIME in h5file, rmb_path
        num_steps = h5file["time"].shape[0]
        data = np.broadcast_to(pb, (num_steps, pb.shape[0])).copy()

        if DataKey.MATERIAL_PROPERTY in h5file:
            existing = h5file[DataKey.MATERIAL_PROPERTY][:]
            if existing.shape == data.shape and np.array_equal(existing, data):
                status = "unchanged"
            else:
                assert overwrite, (
                    f"{rmb_path}: {DataKey.MATERIAL_PROPERTY} already exists "
                    "with a different value; pass --overwrite to replace it"
                )
                del h5file[DataKey.MATERIAL_PROPERTY]
                h5file.create_dataset(DataKey.MATERIAL_PROPERTY, data=data)
                status = "overwritten"
        else:
            h5file.create_dataset(DataKey.MATERIAL_PROPERTY, data=data)
            status = "added"

        dataset = h5file[DataKey.MATERIAL_PROPERTY]
        dataset.attrs["object_id"] = object_id
        dataset.attrs["object_key"] = object_key
        write_wp4_provenance_attrs(dataset, wp4_provenance)

    return status


def main() -> None:
    args = parse_args()
    filenames = find_rmb_files(str(args.dataset_path))
    assert len(filenames) > 0, args.dataset_path

    pb_table, object_key_to_id = load_pb_table(args.checkpoint)
    wp4_provenance = get_wp4_provenance(args.checkpoint)
    object_id_to_key = {
        object_id: object_key for object_key, object_id in object_key_to_id.items()
    }
    episodes = [
        (filename, WrenchPredictor4Dataset.get_object_id(filename))
        for filename in filenames
    ]
    counts = {
        object_id: {"added": 0, "overwritten": 0, "unchanged": 0}
        for object_id in {object_id for _, object_id in episodes}
    }

    for filename, object_id in episodes:
        status = write_constant_pb(
            filename,
            pb_table[object_id],
            object_id,
            object_id_to_key[object_id],
            wp4_provenance,
            args.overwrite,
        )
        counts[object_id][status] += 1
        print(f"[{status}] {filename}")

    for object_id in sorted(counts):
        object_key = object_id_to_key[object_id]
        pb = pb_table[object_id]
        object_counts = counts[object_id]
        print(f"object: {object_key} (id={object_id})")
        print(f"PB: {pb.tolist()}")
        print(
            f"HDF5 key: {DataKey.MATERIAL_PROPERTY}, "
            f"shape per episode: (T, {pb.shape[0]})"
        )
        print(
            "episodes: "
            f"{sum(object_counts.values())} "
            f"(added={object_counts['added']}, "
            f"overwritten={object_counts['overwritten']}, "
            f"unchanged={object_counts['unchanged']})"
        )

    print(f"total: {len(counts)} objects, {len(episodes)} episodes")


if __name__ == "__main__":
    main()
