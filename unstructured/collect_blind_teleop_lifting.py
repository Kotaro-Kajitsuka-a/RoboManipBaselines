import datetime
import glob
import os
import random
import re
import shutil
import subprocess

# Usage:
# python unstructured/collect_blind_teleop_lifting.py

OBJECT_COUNT = 3
TRIALS_PER_OBJECT_FOR_TRAINING = 25
TRIALS_PER_OBJECT_FOR_VALIDATION = 0
VALIDATION_WORLD_SUFFIX_START = 50

EPISODE_NAME_PATTERN = re.compile(
    r"^WrenchPredObject(?P<object_idx>\d+)_world(?P<world_idx>\d+)_"
    r"(?P<episode_idx>\d+)\.rmb$"
)

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(repo_root)

dataset_dir = os.path.join(repo_root, "robo_manip_baselines", "dataset")

# Always append to the same training dataset directory.
session_dataset_dir = os.path.join(
    dataset_dir,
    "DatasetMujocoUR5eLiftingi",
    "training",
)
os.makedirs(session_dataset_dir, exist_ok=True)

object_dataset_dirs = {}
for object_idx in range(OBJECT_COUNT):
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
    """Return True if a successful trajectory for world_idx already exists."""
    pattern = os.path.join(
        object_dataset_dir,
        f"*_world{world_idx}_*.rmb",
    )
    return len(glob.glob(pattern)) > 0


def delete_duplicate_world_episodes(object_dataset_dirs, log_file):
    """Keep episode 000 and delete later episodes for each object/world."""
    episodes_by_world = {}
    episode_count = 0

    for expected_object_idx, object_dataset_dir in object_dataset_dirs.items():
        for entry in os.scandir(object_dataset_dir):
            if not entry.name.endswith(".rmb"):
                continue
            if not entry.is_dir(follow_symlinks=False):
                raise RuntimeError(f"Expected an RMB directory: {entry.path}")

            match = EPISODE_NAME_PATTERN.fullmatch(entry.name)
            if match is None:
                raise RuntimeError(f"Unexpected RMB name: {entry.path}")

            object_idx = int(match.group("object_idx"))
            world_idx = int(match.group("world_idx"))
            episode_idx = int(match.group("episode_idx"))
            if object_idx != expected_object_idx:
                raise RuntimeError(
                    f"Object directory and RMB name disagree: {entry.path}"
                )

            episodes_by_world.setdefault((object_idx, world_idx), []).append(
                (episode_idx, entry.path)
            )
            episode_count += 1

    deleted_count = 0
    for (object_idx, world_idx), episodes in sorted(episodes_by_world.items()):
        if len(episodes) == 1:
            continue

        episodes.sort()
        episode_zero_paths = [
            path for episode_idx, path in episodes if episode_idx == 0
        ]
        if len(episode_zero_paths) != 1:
            raise RuntimeError(
                f"Duplicate world has no unique episode 000: "
                f"object_idx={object_idx}, world_idx={world_idx}, "
                f"episodes={episodes}"
            )

        for episode_idx, path in episodes:
            if episode_idx == 0:
                continue
            shutil.rmtree(path)
            deleted_count += 1
            print(
                f"Delete duplicate episode: object_idx={object_idx}, "
                f"world_idx={world_idx}, episode_idx={episode_idx}, path={path}",
                file=log_file,
                flush=True,
            )

    print(
        f"Checked {episode_count} episode(s) across "
        f"{len(episodes_by_world)} world(s); "
        f"deleted {deleted_count} duplicate episode(s).",
        file=log_file,
        flush=True,
    )


def create_trials():
    assert 0 <= TRIALS_PER_OBJECT_FOR_TRAINING <= VALIDATION_WORLD_SUFFIX_START
    assert (
        0 <= TRIALS_PER_OBJECT_FOR_VALIDATION <= (100 - VALIDATION_WORLD_SUFFIX_START)
    )

    trials = []
    for object_idx in range(OBJECT_COUNT):
        env_name = f"MujocoUR5eLiftingi_I{object_idx}"
        world_idx_base = object_idx * 100

        for world_suffix in range(TRIALS_PER_OBJECT_FOR_TRAINING):
            trials.append(
                ("training", object_idx, env_name, world_idx_base + world_suffix)
            )

        for validation_idx in range(TRIALS_PER_OBJECT_FOR_VALIDATION):
            world_suffix = VALIDATION_WORLD_SUFFIX_START + validation_idx
            trials.append(
                ("validation", object_idx, env_name, world_idx_base + world_suffix)
            )

    return trials


trials = create_trials()

# Hide whether the current trial uses I0, I1, or I2 from the operator.
random.shuffle(trials)

log_dir = os.path.join(
    dataset_dir,
    "blind_teleop_lifting_logs",
)
os.makedirs(log_dir, exist_ok=True)

session_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = os.path.join(
    log_dir,
    f"session_{session_datetime}.log",
)

with open(log_path, "w") as log_file:
    print(f"dataset_dir={session_dataset_dir}", file=log_file, flush=True)
    print(
        f"training_trials_per_object={TRIALS_PER_OBJECT_FOR_TRAINING}, "
        f"validation_trials_per_object={TRIALS_PER_OBJECT_FOR_VALIDATION}, "
        f"validation_world_suffix_start={VALIDATION_WORLD_SUFFIX_START}",
        file=log_file,
        flush=True,
    )

    delete_duplicate_world_episodes(object_dataset_dirs, log_file)

    for trial_idx, (split_name, object_idx, env_name, world_idx) in enumerate(trials):
        demo_name = f"WrenchPredObject{object_idx}"
        object_dataset_dir = object_dataset_dirs[object_idx]

        if world_idx_exists(object_dataset_dir, world_idx):
            print(
                f"[{trial_idx + 1}/{len(trials)}] Skip an already collected trial."
            )
            print(
                f"[{trial_idx + 1}/{len(trials)}] Skip split={split_name}, "
                f"env={env_name}, world_idx={world_idx} (already exists)",
                file=log_file,
                flush=True,
            )
            continue

        print(f"[{trial_idx + 1}/{len(trials)}] Start the next trial.")
        print(
            f"Trial {trial_idx + 1}/{len(trials)}: "
            f"split={split_name}, env={env_name}, world_idx={world_idx}",
            file=log_file,
            flush=True,
        )

        command = [
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
            source_path = os.path.join(teleop_output_dir, saved_data_name)
            destination_path = os.path.join(object_dataset_dir, saved_data_name)

            if os.path.exists(destination_path):
                raise FileExistsError(f"{destination_path} already exists.")

            shutil.move(source_path, destination_path)
            moved_count += 1

        os.rmdir(teleop_output_dir)
        print(
            f"Moved {moved_count} data item(s) to {object_dataset_dir}",
            file=log_file,
            flush=True,
        )

    delete_duplicate_world_episodes(object_dataset_dirs, log_file)

print(f"Finished. Dataset: {session_dataset_dir}")
print(f"Log: {log_path}")
