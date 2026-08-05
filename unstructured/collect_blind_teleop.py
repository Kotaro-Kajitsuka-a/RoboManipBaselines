import datetime
import glob
import os
import random
import shutil
import subprocess

# Usage:
# uv run python unstructured/collect_blind_teleop.py

TRIALS_PER_OBJECT_FOR_TRAINING = 20
# TODO: Add validation data collection.

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(repo_root)

dataset_dir = os.path.join(repo_root, "robo_manip_baselines", "dataset")

# ---------------------------------------------------------------------
# Always append to the same dataset directory.
# ---------------------------------------------------------------------
session_dataset_dir = os.path.join(
    dataset_dir,
    "DatasetMujocoXarm7Pusht",
)
os.makedirs(session_dataset_dir, exist_ok=True)

object_dataset_dirs = {}
for object_idx in range(5):
    object_dataset_dir = os.path.join(
        session_dataset_dir,
        f"WrenchPredObject{object_idx}",
    )
    os.makedirs(object_dataset_dir, exist_ok=True)
    object_dataset_dirs[object_idx] = object_dataset_dir


def find_teleop_output_dirs(demo_name):
    """Find temporary Teleop output directories."""
    return {
        entry.path
        for entry in os.scandir(dataset_dir)
        if entry.is_dir() and entry.name.startswith(f"{demo_name}_")
    }


def world_idx_exists(object_dataset_dir, world_idx):
    """
    Return True if a successful trajectory for world_idx already exists.

    Example filename:
        WrenchPredObject4_world416_000.rmb
    """
    pattern = os.path.join(
        object_dataset_dir,
        f"*_world{world_idx}_*.rmb",
    )
    return len(glob.glob(pattern)) > 0


# ---------------------------------------------------------------------
# Create all trials.
# ---------------------------------------------------------------------
trials = []
for object_idx in range(5):
    for world_suffix in range(TRIALS_PER_OBJECT_FOR_TRAINING):
        env_name = f"MujocoXarm7Pusht_T{object_idx}"
        world_idx = object_idx * 100 + world_suffix
        trials.append((env_name, world_idx))

# Hide object order from the operator.
random.shuffle(trials)

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
log_dir = os.path.join(
    dataset_dir,
    "blind_teleop_logs",
)
os.makedirs(log_dir, exist_ok=True)

session_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = os.path.join(
    log_dir,
    f"session_{session_datetime}.log",
)

with open(log_path, "w") as log_file:
    print(
        f"dataset_dir={session_dataset_dir}",
        file=log_file,
        flush=True,
    )

    for trial_idx, (env_name, world_idx) in enumerate(trials):
        object_idx = world_idx // 100
        demo_name = f"WrenchPredObject{object_idx}"
        object_dataset_dir = object_dataset_dirs[object_idx]

        # -------------------------------------------------------------
        # Skip if this world has already been collected.
        # -------------------------------------------------------------
        if world_idx_exists(object_dataset_dir, world_idx):
            msg = (
                f"[{trial_idx + 1}/{len(trials)}] "
                f"Skip world_idx={world_idx} "
                f"(already exists)"
            )
            print(msg)
            print(msg, file=log_file, flush=True)
            continue

        print(
            f"Trial {trial_idx + 1}/{len(trials)}: "
            f"env={env_name}, world_idx={world_idx}",
            file=log_file,
            flush=True,
        )

        command = [
            "uv",
            "run",
            "python",
            "robo_manip_baselines/bin/Teleop.py",
            env_name,
            "--demo_name",
            demo_name,
            "--world_idx_list",
            str(world_idx),
        ]

        output_dirs_before = find_teleop_output_dirs(demo_name)

        subprocess.run(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=True,
        )

        output_dirs_after = find_teleop_output_dirs(demo_name)
        new_output_dirs = output_dirs_after - output_dirs_before

        if len(new_output_dirs) != 1:
            raise RuntimeError(
                f"Expected one new Teleop output directory for "
                f"{demo_name}, but found "
                f"{len(new_output_dirs)}: "
                f"{sorted(new_output_dirs)}"
            )

        teleop_output_dir = new_output_dirs.pop()

        saved_data_names = os.listdir(teleop_output_dir)
        if not saved_data_names:
            raise RuntimeError(f"No data was saved in {teleop_output_dir}")

        moved_count = 0

        for saved_data_name in saved_data_names:
            source_path = os.path.join(
                teleop_output_dir,
                saved_data_name,
            )

            destination_path = os.path.join(
                object_dataset_dir,
                saved_data_name,
            )

            if os.path.exists(destination_path):
                raise FileExistsError(f"{destination_path} already exists.")

            shutil.move(
                source_path,
                destination_path,
            )

            moved_count += 1

        os.rmdir(teleop_output_dir)

        print(
            f"Moved {moved_count} data item(s) to " f"{object_dataset_dir}",
            file=log_file,
            flush=True,
        )

print(f"Finished. Dataset: {session_dataset_dir}")
print(f"Log: {log_path}")
