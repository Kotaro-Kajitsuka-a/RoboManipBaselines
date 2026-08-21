import unittest

import numpy as np
import torch

from robo_manip_baselines.common import DataKey
from robo_manip_baselines.policy.wrench_predictor4_online.WrenchPredictor4OnlineUtils import (
    ONLINE_PB_STD_KEY,
    GaussianBeliefOnlinePb,
    resolve_gaussian_num_points,
)


class TestGaussianBeliefOnlinePb(unittest.TestCase):
    def test_online_pb_std_is_available_as_policy_state(self):
        self.assertEqual(ONLINE_PB_STD_KEY, DataKey.ONLINE_PB_STD)
        self.assertIn(DataKey.ONLINE_PB_STD, DataKey.MEASURED_DATA_KEYS)
        self.assertEqual(DataKey.get_dim(DataKey.ONLINE_PB_STD, env=None), 1)

    def test_default_quadrature_order_scales_with_pb_dimension(self):
        self.assertEqual(resolve_gaussian_num_points(1, None), 16)
        self.assertEqual(resolve_gaussian_num_points(2, None), 32)
        self.assertEqual(resolve_gaussian_num_points(1, 9), 9)

    def test_even_quadrature_order(self):
        belief = GaussianBeliefOnlinePb(
            initial_mean=np.array([0.2], dtype=np.float32),
            initial_std=0.25,
            num_points=64,
            beta=10.0,
            device=torch.device("cpu"),
        )

        self.assertEqual(belief.get_candidates().shape, (64, 1))
        weights = belief.update(torch.zeros(64))
        self.assertEqual(weights.shape, (64,))
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=6)

    def test_flat_loss_preserves_gaussian(self):
        belief = GaussianBeliefOnlinePb(
            np.array([0.2], dtype=np.float32),
            initial_std=0.3,
            num_points=9,
            beta=100.0,
            device=torch.device("cpu"),
        )

        initial_mean = belief.mean.clone()
        initial_std = belief.std.clone()
        belief.update(torch.full((belief.num_points,), 0.7))

        torch.testing.assert_close(belief.mean, initial_mean, atol=1e-6, rtol=0.0)
        torch.testing.assert_close(belief.std, initial_std, atol=1e-6, rtol=0.0)

    def test_informative_loss_moves_and_narrows_gaussian(self):
        belief = GaussianBeliefOnlinePb(
            np.array([0.0], dtype=np.float32),
            initial_std=0.4,
            num_points=9,
            beta=20.0,
            device=torch.device("cpu"),
        )

        candidates = belief.get_candidates().squeeze(1)
        losses = (candidates - 0.3).square()
        belief.update(losses)

        self.assertGreater(belief.mean.item(), 0.0)
        self.assertLess(abs(belief.mean.item() - 0.3), 0.3)
        self.assertLess(belief.std.item(), 0.4)

    def test_candidates_have_expected_shape(self):
        belief = GaussianBeliefOnlinePb(
            np.array([0.1], dtype=np.float32),
            initial_std=0.2,
            num_points=15,
            beta=1.0,
            device=torch.device("cpu"),
        )

        self.assertEqual(belief.get_candidates().shape, (15, 1))


if __name__ == "__main__":
    unittest.main()
