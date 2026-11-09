import math
import unittest

import torch

from numerical_monitor.metrics import tensor_metrics


class TensorMetricsTest(unittest.TestCase):
    def test_finite_population_and_fault_counts(self):
        x = torch.tensor(
            [0.0, 3.0, 4.0, float("nan"), float("inf"), -float("inf")], requires_grad=True
        )
        before = x.detach().clone()
        stats = tensor_metrics(x)
        self.assertEqual(stats["numel"], 6)
        self.assertEqual(stats["finite_count"], 3)
        self.assertAlmostEqual(stats["mean"], 7 / 3, places=6)
        self.assertAlmostEqual(
            stats["std"], torch.tensor([0.0, 3.0, 4.0]).std(correction=0).item(), places=6
        )
        self.assertAlmostEqual(stats["rms"], math.sqrt(25 / 3), places=6)
        self.assertAlmostEqual(stats["l2_norm"], 5.0)
        self.assertAlmostEqual(stats["zero_ratio"], 1 / 3, places=6)
        self.assertEqual(stats["nan_count"], 1)
        self.assertEqual(stats["inf_count"], 2)
        self.assertEqual(stats["posinf_count"], 1)
        self.assertEqual(stats["neginf_count"], 1)
        torch.testing.assert_close(x.detach(), before, equal_nan=True)
        self.assertIsNone(x.grad)

    def test_empty_all_invalid_and_singleton(self):
        for x in (torch.empty(0), torch.tensor([float("nan"), float("inf")])):
            stats = tensor_metrics(x)
            self.assertEqual(stats["finite_count"], 0)
            self.assertTrue(math.isnan(stats["mean"]))
            self.assertEqual(stats["l2_norm"], 0)
        stats = tensor_metrics(torch.tensor([7.0]))
        self.assertEqual(stats["std"], 0)
        self.assertEqual(stats["rms"], 7)

    def test_large_finite_rms_and_standard_deviation(self):
        stats = tensor_metrics(torch.tensor([-1e20, 1e20]))
        self.assertTrue(math.isfinite(stats["rms"]))
        self.assertTrue(math.isfinite(stats["std"]))
        self.assertAlmostEqual(stats["std"] / 1e20, 1, places=5)
        self.assertAlmostEqual(stats["rms"] / 1e20, 1, places=5)

    def test_rejects_complex(self):
        with self.assertRaises(TypeError):
            tensor_metrics(torch.tensor([1 + 2j]))

    def test_subnormal_values_are_not_rescaled_twice(self):
        x = torch.tensor([1e-40], dtype=torch.float32)
        stats = tensor_metrics(x)
        for key in ("mean", "rms", "l2_norm"):
            self.assertEqual(stats[key], x.item())
