import datetime
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
session_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
session_dataset_dir = os.path.join(
    dataset_dir,
    f"DatasetMujocoXarm7Pusht_{session_datetime}",
)
os.makedirs(session_dataset_dir)

object_dataset_dirs = {}
for object_idx in range(5):
    object_dataset_dir = os.path.join(
        session_dataset_dir, f"WrenchPredObject{object_idx}"
    )
    os.makedirs(object_dataset_dir)
    object_dataset_dirs[object_idx] = object_dataset_dir


def find_teleop_output_dirs(demo_name):
    return {
        entry.path
        for entry in os.scandir(dataset_dir)
        if entry.is_dir() and entry.name.startswith(f"{demo_name}_")
    }


# Select world indexes without duplication for each object.
trials = []
for object_idx in range(5):
    for world_suffix in range(TRIALS_PER_OBJECT_FOR_TRAINING):
        env_name = f"MujocoXarm7Pusht_T{object_idx}"
        world_idx = 100 * object_idx + world_suffix
        trials.append((env_name, world_idx))

# Hide the order of the objects from the operator.
random.shuffle(trials)

log_dir = "robo_manip_baselines/dataset/blind_teleop_logs"
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, f"session_{session_datetime}.log")

with open(log_path, "w") as log_file:
    print(f"dataset_dir={session_dataset_dir}", file=log_file, flush=True)

    for trial_idx, (env_name, world_idx) in enumerate(trials):
        object_idx = world_idx // 100
        demo_name = f"WrenchPredObject{object_idx}"
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
        output_dirs_before_teleop = find_teleop_output_dirs(demo_name)
        subprocess.run(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=True,
        )

        output_dirs_after_teleop = find_teleop_output_dirs(demo_name)
        new_output_dirs = output_dirs_after_teleop - output_dirs_before_teleop
        if len(new_output_dirs) != 1:
            raise RuntimeError(
                f"Expected one new Teleop output directory for {demo_name}, "
                f"but found {len(new_output_dirs)}: {sorted(new_output_dirs)}"
            )

        teleop_output_dir = new_output_dirs.pop()
        saved_data_names = os.listdir(teleop_output_dir)
        if not saved_data_names:
            raise RuntimeError(f"No data was saved in {teleop_output_dir}")

        object_dataset_dir = object_dataset_dirs[object_idx]
        for saved_data_name in saved_data_names:
            source_path = os.path.join(teleop_output_dir, saved_data_name)
            destination_path = os.path.join(object_dataset_dir, saved_data_name)
            if os.path.exists(destination_path):
                raise FileExistsError(destination_path)
            shutil.move(source_path, destination_path)
        os.rmdir(teleop_output_dir)

        print(
            f"Moved {len(saved_data_names)} data item(s) to {object_dataset_dir}",
            file=log_file,
            flush=True,
        )

        # After saving an episode, press Esc in Teleop.
        # subprocess.run() then returns and the loop starts the next environment.
