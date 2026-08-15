import unittest

import torch

from robo_manip_baselines.policy.wrench_predictor5.WrenchPredictor5Model import (
    WrenchPredictor5Model,
)


class TestWrenchPredictor5Model(unittest.TestCase):
    def test_forward_and_backward_shapes(self):
        batch_size = 2
        horizon = 4
        n_obs_steps = 2
        image_feature_dim = 16 * 15 * 20
        model = WrenchPredictor5Model(
            image_feature_dim=image_feature_dim,
            state_dim=10,
            action_dim=10,
            wrench_dim=6,
            num_objects=7,
            pb_dim=1,
            horizon=horizon,
            n_obs_steps=n_obs_steps,
            latent_shape=(16, 15, 20),
            hidden_dim=32,
            nhead=4,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dim_feedforward=64,
            dropout=0.0,
            wrench_loss_weight=0.1,
        )
        batch = {
            "image_feature": torch.randn(
                batch_size,
                horizon,
                image_feature_dim,
            ),
            "state": torch.randn(batch_size, horizon, 10),
            "action": torch.randn(batch_size, horizon, 10),
            "wrench": torch.randn(batch_size, horizon, 6),
            "object_id": torch.tensor([0, 1]),
        }

        memory = model.get_encoder_memory(batch)
        num_encoder_tokens = (
            n_obs_steps * 16 + n_obs_steps + (horizon - n_obs_steps) + 1
        )
        self.assertEqual(memory.shape, (batch_size, num_encoder_tokens, 32))
        queries = model.get_decoder_queries(batch_size)
        num_decoder_tokens = (horizon - n_obs_steps) * (16 + 1)
        self.assertEqual(queries.shape, (batch_size, num_decoder_tokens, 32))

        pred = model(batch)
        self.assertEqual(
            pred["image_feature"].shape,
            (batch_size, horizon - n_obs_steps, image_feature_dim),
        )
        self.assertEqual(
            pred["wrench"].shape,
            (batch_size, horizon - n_obs_steps, 6),
        )

        result = model.compute_loss(batch)
        result["loss"].backward()
        self.assertIsNotNone(model.material_property.weight.grad)
        self.assertIsNotNone(model.image_feature_proj.weight.grad)
        self.assertIsNotNone(model.latent_output_proj.weight.grad)


if __name__ == "__main__":
    unittest.main()
