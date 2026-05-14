import numpy as np
import torch

from robo_manip_baselines.common import (
    DataKey,
    RmbData,
    denormalize_data,
    get_skipped_data_seq,
)
from robo_manip_baselines.policy.wrench_predictor2.EvalWrenchPredictor import (
    EvalWrenchPredictor as EvalWrenchPredictorBase,
    parse_argument,
)
from robo_manip_baselines.policy.wrench_predictor3.WrenchPredictorPolicy import (
    WrenchPredictorPolicy,
)


class EvalWrenchPredictor(EvalWrenchPredictorBase):
    def setup_image_transform(self):
        pass

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

    def evaluate(self):
        with RmbData(self.rmb_filename) as rmb_data:
            skip = self.model_meta_info["data"]["skip"]
            time_seq = np.asarray(rmb_data[DataKey.TIME][::skip], dtype=np.float64)
            gt_wrench_seq = np.asarray(
                get_skipped_data_seq(
                    rmb_data[DataKey.MEASURED_EEF_WRENCH_MOVING_AVERAGE][:],
                    DataKey.MEASURED_EEF_WRENCH_MOVING_AVERAGE,
                    skip,
                ),
                dtype=np.float64,
            )
            gt_wrench_seq = self.get_percentile_clipped_wrench(gt_wrench_seq)
            assert len(time_seq) >= 2
            valid_episode_len = len(time_seq) - 1

            pred_wrench_list = []
            for time_idx in range(valid_episode_len):
                state = self.build_state(rmb_data, time_idx)
                with torch.inference_mode():
                    pred_wrench_chunk = (
                        self.policy(
                            state,
                            material_property=self.material_property_tensor,
                        )[0]
                        .detach()
                        .cpu()
                        .numpy()
                    )
                pred_wrench = denormalize_data(
                    pred_wrench_chunk[0], self.model_meta_info["action"]
                )
                pred_wrench_list.append(pred_wrench)

        pred_wrench_seq = np.asarray(pred_wrench_list, dtype=np.float64)
        time_seq = time_seq[1:]
        gt_wrench_seq = gt_wrench_seq[1:]
        time_interval_seq = np.diff(time_seq)
        print(
            f"[{self.__class__.__name__}] time interval: "
            f"mean={time_interval_seq.mean():.6f} [s], "
            f"std={time_interval_seq.std():.6f} [s]"
        )
        abs_error_seq = np.abs(pred_wrench_seq - gt_wrench_seq)
        mae = np.mean(abs_error_seq, axis=0)
        return time_seq, gt_wrench_seq, pred_wrench_seq, abs_error_seq, mae


if __name__ == "__main__":
    evaluator = EvalWrenchPredictor(**vars(parse_argument()))
    evaluator.run()
