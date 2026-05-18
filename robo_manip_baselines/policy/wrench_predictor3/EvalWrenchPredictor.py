import argparse
import os
import pickle

import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.transforms import v2

from robo_manip_baselines.common import (
    DataKey,
    RmbData,
    denormalize_data,
    find_rmb_files,
    normalize_data,
)
from robo_manip_baselines.policy.wrench_predictor3.WrenchPredictorPolicy import (
    WrenchPredictorPolicy,
)
from robo_manip_baselines.policy.wrench_predictor3.MaterialPropertyUtils import (
    extract_material_object_key,
)
from robo_manip_baselines.policy.wrench_predictor3.WrenchPredictorSequenceUtils import (
    build_condition,
)


def parse_material_property(material_property):
    material_property = (
        material_property.replace("[", " ")
        .replace("]", " ")
        .replace(",", " ")
        .replace("\u00a0", " ")
    )
    return np.asarray(material_property.split(), dtype=np.float32)


def parse_argument():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("checkpoint", type=str, help="checkpoint file")
    parser.add_argument(
        "rmb_path",
        type=str,
        help="path to one RMB episode file or a directory containing exactly one RMB file",
    )
    parser.add_argument(
        "--material_property",
        type=str,
        default=None,
        help='fixed material property vector, e.g. "0.1 -0.2 ..."',
    )
    parser.add_argument(
        "--material_object_key",
        type=str,
        default=None,
        help="object key used to select material property from checkpoint (extracted from rmb_path by default)",
    )
    return parser.parse_args()


class EvalWrenchPredictor:
    def __init__(
        self,
        checkpoint,
        rmb_path,
        material_property=None,
        material_object_key=None,
    ):
        self.checkpoint = checkpoint
        self.rmb_path = rmb_path
        self.material_property_arg = material_property
        self.material_object_key = material_object_key

        self.setup_paths()
        self.setup_model_meta_info()
        self.setup_policy()
        self.setup_material_property()
        self.setup_image_transform()

    def setup_paths(self):
        rmb_path_list = find_rmb_files(self.rmb_path)
        if len(rmb_path_list) != 1:
            raise ValueError(
                f"[{self.__class__.__name__}] Expected exactly one RMB file, got {len(rmb_path_list)}: {self.rmb_path}"
            )

        self.checkpoint_dir = os.path.dirname(self.checkpoint)
        self.set_rmb_filename(rmb_path_list[0])

    def set_rmb_filename(self, rmb_filename):
        self.rmb_filename = rmb_filename
        checkpoint_stem = os.path.splitext(os.path.basename(self.checkpoint))[0]
        rmb_stem = os.path.basename(self.rmb_filename.rstrip("/")).replace(".rmb", "")
        output_dir = getattr(self, "output_dir", self.checkpoint_dir)
        self.output_png = os.path.join(
            output_dir, f"{rmb_stem}_{checkpoint_stem}_eval.png"
        )

    def setup_model_meta_info(self):
        model_meta_info_path = os.path.join(self.checkpoint_dir, "model_meta_info.pkl")
        with open(model_meta_info_path, "rb") as f:
            self.model_meta_info = pickle.load(f)
        print(
            f"[{self.__class__.__name__}] Load model meta info: {model_meta_info_path}"
        )

    def setup_material_property(self):
        dim = self.model_meta_info["material_property"]["dim"]
        if self.material_property_arg is not None:
            self.material_property = parse_material_property(self.material_property_arg)
            if self.material_property.shape != (dim,):
                raise ValueError(
                    f"--material_property must have {dim} values, "
                    f"got {self.material_property.shape[0]}: {self.material_property_arg}"
                )
            self.set_material_property_tensor()
            return

        object_key = self.material_object_key
        if object_key is None:
            object_key = extract_material_object_key(self.rmb_filename)
        assert object_key is not None, self.rmb_filename

        object_id = self.model_meta_info["material_property"]["object_key_to_id"][
            object_key
        ]
        self.material_property = (
            self.policy.material_property_embedding.weight[object_id]
            .detach()
            .cpu()
            .numpy()
        )
        self.set_material_property_tensor()

    def set_material_property_tensor(self):
        self.material_property_tensor = torch.tensor(
            self.material_property[np.newaxis],
            dtype=torch.float32,
            device=self.device,
        )
        print(
            f"[{self.__class__.__name__}] material_property: {self.material_property}"
        )

    def setup_policy(self):
        state_dim = len(self.model_meta_info["state"]["example"])
        wrench_dim = len(self.model_meta_info["action"]["example"])
        material_property_dim = self.model_meta_info["material_property"]["dim"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = WrenchPredictorPolicy(
            state_dim=state_dim,
            wrench_dim=wrench_dim,
            num_material_objects=len(
                self.model_meta_info["material_property"]["object_key_to_id"]
            ),
            material_property_dim=material_property_dim,
            policy_args=self.model_meta_info["policy"]["args"],
        )
        print(f"[{self.__class__.__name__}] Load checkpoint: {self.checkpoint}")
        self.policy.load_state_dict(
            torch.load(self.checkpoint, map_location=self.device, weights_only=True)
        )
        self.policy.to(self.device)
        self.policy.eval()

    def setup_image_transform(self):
        self.image_transform = v2.Compose([v2.ToDtype(torch.float32, scale=True)])

    def get_percentile_clipped_wrench(self, wrench):
        clip_info = self.model_meta_info["action"]["percentile_clip"]
        return np.clip(wrench, clip_info["min"], clip_info["max"])

    def build_condition_tensor(self, rmb_data, center_time_idx):
        horizon = self.model_meta_info["data"]["horizon"]
        condition = build_condition(
            rmb_data,
            self.model_meta_info["state"]["keys"],
            self.model_meta_info["action"]["keys"],
            center_time_idx,
            horizon,
        )
        condition = normalize_data(condition, self.model_meta_info["state"])
        return torch.tensor(
            condition[np.newaxis], dtype=torch.float32, device=self.device
        )

    def build_images(self, rmb_data, center_time_idx):
        horizon = self.model_meta_info["data"]["horizon"]
        image_time_idxes = np.array([center_time_idx - horizon, center_time_idx])
        image_keys = [
            DataKey.get_rgb_image_key(camera_name)
            for camera_name in self.model_meta_info["image"]["camera_names"]
        ]
        images = np.stack(
            [
                np.stack(
                    [
                        rmb_data[key][int(image_time_idx)]
                        for image_time_idx in image_time_idxes
                    ],
                    axis=0,
                )
                for key in image_keys
            ],
            axis=0,
        )
        images = np.moveaxis(images, -1, -3)
        images_tensor = torch.tensor(images, dtype=torch.uint8)
        images_tensor = self.image_transform(images_tensor).unsqueeze(0).to(self.device)
        return images_tensor

    def evaluate(self):
        image_size = self.model_meta_info["data"]["image_size"]
        with RmbData(self.rmb_filename, image_size=image_size) as rmb_data:
            horizon = self.model_meta_info["data"]["horizon"]
            time_seq = np.asarray(rmb_data[DataKey.TIME][:], dtype=np.float64)
            gt_wrench_seq = self.get_percentile_clipped_wrench(
                np.asarray(
                    rmb_data[DataKey.MEASURED_EEF_WRENCH_MOVING_AVERAGE][:],
                    dtype=np.float64,
                )
            )
            assert len(time_seq) >= 2 * horizon + 1

            target_time_list = []
            gt_wrench_list = []
            pred_wrench_list = []
            center_time_idxes = range(horizon, len(time_seq) - horizon, horizon)
            for center_time_idx in center_time_idxes:
                condition = self.build_condition_tensor(rmb_data, center_time_idx)
                images = self.build_images(rmb_data, center_time_idx)
                with torch.inference_mode():
                    pred_wrench_chunk = (
                        self.policy(
                            condition,
                            images,
                            material_property=self.material_property_tensor,
                        )[0]
                        .detach()
                        .cpu()
                        .numpy()
                    )
                pred_wrench_chunk = denormalize_data(
                    pred_wrench_chunk, self.model_meta_info["action"]
                )
                target_slice = slice(center_time_idx + 1, center_time_idx + horizon + 1)
                target_time_list.append(time_seq[target_slice])
                gt_wrench_list.append(gt_wrench_seq[target_slice])
                pred_wrench_list.append(pred_wrench_chunk)

        time_seq = np.concatenate(target_time_list, axis=0)
        gt_wrench_seq = np.concatenate(gt_wrench_list, axis=0)
        pred_wrench_seq = np.concatenate(pred_wrench_list, axis=0)
        if len(time_seq) > 1:
            time_interval_seq = np.diff(time_seq)
            print(
                f"[{self.__class__.__name__}] time interval: "
                f"mean={time_interval_seq.mean():.6f} [s], "
                f"std={time_interval_seq.std():.6f} [s]"
            )
        else:
            print(
                f"[{self.__class__.__name__}] time interval: "
                "not available for a single target sample"
            )
        abs_error_seq = np.abs(pred_wrench_seq - gt_wrench_seq)
        mae = np.mean(abs_error_seq, axis=0)
        return time_seq, gt_wrench_seq, pred_wrench_seq, abs_error_seq, mae

    def run(self):
        time_seq, gt_wrench_seq, pred_wrench_seq, _abs_error_seq, mae = self.evaluate()
        self.save_plot(time_seq, gt_wrench_seq, pred_wrench_seq, mae)
        self.print_metrics(mae)

    def save_plot(self, time_seq, gt_wrench_seq, pred_wrench_seq, mae):
        labels = ["Fx", "Fy", "Fz", "Nx", "Ny", "Nz"]
        unit_list = ["N"] * 3 + ["Nm"] * 3

        fig, axes = plt.subplots(
            6, 1, figsize=(12, 14), sharex=True, constrained_layout=True
        )
        fig.suptitle(
            f"Wrench prediction: {os.path.basename(self.rmb_filename)}", fontsize=16
        )

        for axis_idx, ax in enumerate(axes):
            ax.plot(
                time_seq,
                gt_wrench_seq[:, axis_idx],
                label="gt",
                linewidth=2.0,
            )
            ax.plot(
                time_seq,
                pred_wrench_seq[:, axis_idx],
                label="pred",
                linewidth=1.5,
                alpha=0.85,
            )
            ax.set_title(f"{labels[axis_idx]}  MAE={mae[axis_idx]:.4f}")
            ax.set_ylabel(f"{labels[axis_idx]} [{unit_list[axis_idx]}]")
            ax.grid(True)
            ax.legend(loc="upper right")

        axes[-1].set_xlabel("Time [s]")
        fig.savefig(self.output_png, dpi=150)
        plt.close(fig)
        print(f"[{self.__class__.__name__}] Save plot: {self.output_png}")

    def print_metrics(self, mae):
        labels = ["Fx", "Fy", "Fz", "Nx", "Ny", "Nz"]
        force_xy_mae = mae[:2].mean()
        torque_mae = mae[3:].mean()

        print(f"[{self.__class__.__name__}] MAE:")
        for label, value in zip(labels, mae):
            print(f"  - {label}: {value:.6f}")
        print(f"  - Fz only: {mae[2]:.6f}")
        print(f"  - Fx/Fy mean: {force_xy_mae:.6f}")
        print(f"  - torque mean: {torque_mae:.6f}")


if __name__ == "__main__":
    evaluator = EvalWrenchPredictor(**vars(parse_argument()))
    evaluator.run()
