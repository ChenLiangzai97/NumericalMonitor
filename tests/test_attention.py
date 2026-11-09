"""用可解析结果验证 Attention 统计，覆盖 mask 与采样边界条件。"""

import math
import unittest

import torch

from numerical_monitor.attention import AttentionMonitor


class AttentionTests(unittest.TestCase):
    def test_uniform_and_per_head_norms(self):
        monitor = AttentionMonitor(log_interval=1, query_chunk_size=1)
        q = torch.zeros(2, 2, 3, 4, requires_grad=True)
        k = torch.ones(2, 2, 5, 4, requires_grad=True)
        result = monitor.observe("sa", q, k, step=0)
        metrics = result["per_head"]
        for value in metrics["entropy"]:
            self.assertAlmostEqual(value, math.log(5), places=6)
        self.assertEqual(metrics["q_norm"], [0.0, 0.0])
        self.assertEqual(metrics["k_norm"], [2.0, 2.0])
        self.assertEqual(metrics["valid_query_count"], [6, 6])
        self.assertEqual(metrics["logit_max"], [0.0, 0.0])
        self.assertIsNone(q.grad)
        self.assertIsNone(k.grad)

    def test_masked_rows_and_additive_mask(self):
        q = torch.zeros(1, 1, 2, 4)
        k = torch.zeros(1, 1, 3, 4)
        mask = torch.tensor([[True, False, False], [False, False, False]])
        monitor = AttentionMonitor(log_interval=1)
        result = monitor.observe("sa", q, k, 0, attn_mask=mask)["per_head"]
        self.assertEqual(result["entropy"], [0.0])
        self.assertEqual(result["valid_query_count"], [1])
        self.assertEqual(result["fully_masked_query_count"], [1])
        additive = torch.where(mask, 0.0, -torch.inf)
        other = monitor.observe("sa", q, k, 0, attn_mask=additive)["per_head"]
        self.assertEqual(result, other)

    def test_all_masked_has_no_entropy_or_max(self):
        logits = torch.full((1, 1, 2, 3), -torch.inf)
        result = AttentionMonitor(log_interval=1).observe_logits("sa", logits, 0)["per_head"]
        self.assertTrue(math.isnan(result["entropy"][0]))
        self.assertTrue(math.isnan(result["logit_max"][0]))
        self.assertEqual(result["inf_count"], [0])
        self.assertEqual(result["fully_masked_query_count"], [2])

    def test_causal_sample_uses_original_positions(self):
        monitor = AttentionMonitor(log_interval=1, max_query_samples=2, query_chunk_size=1)
        q = torch.zeros(1, 1, 4, 2)
        k = torch.zeros(1, 1, 4, 2)
        result = monitor.observe("sa", q, k, 0, is_causal=True)
        self.assertTrue(result["sampled"])
        self.assertAlmostEqual(result["per_head"]["entropy"][0], math.log(4) / 2, places=6)

    def test_scaled_logits_match_recomputed(self):
        torch.manual_seed(10)
        q, k = torch.randn(2, 3, 7, 4), torch.randn(2, 3, 5, 4)
        monitor = AttentionMonitor(log_interval=1, query_chunk_size=2)
        observed = monitor.observe("sa", q, k, 0, scale=0.7)["per_head"]
        supplied = monitor.observe_logits("sa", (q @ k.transpose(-1, -2)) * 0.7, 0)["per_head"]
        for name in supplied:
            torch.testing.assert_close(torch.tensor(observed[name]), torch.tensor(supplied[name]))

    def test_outer_autocast_does_not_downcast_recompute(self):
        torch.manual_seed(42)
        q, k = torch.randn(2, 3, 7, 4), torch.randn(2, 3, 5, 4)
        monitor = AttentionMonitor(log_interval=1, query_chunk_size=2)
        expected = monitor.observe("sa", q, k, 0)["per_head"]
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            actual = monitor.observe("sa", q, k, 0)["per_head"]
        for name in expected:
            torch.testing.assert_close(
                torch.tensor(actual[name]),
                torch.tensor(expected[name]),
                rtol=0,
                atol=0,
            )

    def test_gqa_is_explicit(self):
        q, k = torch.ones(1, 4, 3, 2), torch.ones(1, 2, 5, 2)
        monitor = AttentionMonitor(log_interval=1)
        with self.assertRaises(ValueError):
            monitor.observe("sa", q, k, 0)
        result = monitor.observe("sa", q, k, 0, enable_gqa=True)
        self.assertEqual(result["key_heads"], 2)
        self.assertEqual(len(result["per_head"]["k_norm"]), 4)

    def test_numeric_faults_exclude_rows(self):
        logits = torch.tensor([[[[0.0, 0.0], [torch.nan, 1.0], [torch.inf, 0.0]]]])
        result = AttentionMonitor(log_interval=1).observe_logits("sa", logits, 0)["per_head"]
        self.assertAlmostEqual(result["entropy"][0], math.log(2), places=6)
        self.assertEqual(result["nan_count"], [1])
        self.assertEqual(result["inf_count"], [1])
        self.assertEqual(result["invalid_query_count"], [2])
        self.assertEqual(result["logit_max"], [1.0])

    def test_logging_and_interval(self):
        records = []
        monitor = AttentionMonitor(lambda *args: records.append(args), log_interval=5)
        q = k = torch.ones(1, 2, 3, 4)
        self.assertIsNone(monitor.observe("test", q, k, 1))
        monitor.observe("test", q, k, 5)
        self.assertIn(("Attention/test/head_0/q_norm", 2.0, 5), records)

    def test_validation(self):
        with self.assertRaises(ValueError):
            AttentionMonitor(query_chunk_size=0)
        q = k = torch.ones(1, 2, 3, 4)
        monitor = AttentionMonitor(log_interval=1)
        with self.assertRaises(ValueError):
            monitor.observe("sa", q, k, 0, attn_mask=torch.ones(9, 9, dtype=torch.bool))
        with self.assertRaises(ValueError):
            monitor.observe("sa", q, k, 0, scale=float("inf"))


if __name__ == "__main__":
    unittest.main()
