import csv
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch
from torch import nn

from numerical_monitor.gradient_monitor import GradientMonitor


def value(monitor, phase, layer, metric):
    return [
        r["value"]
        for r in monitor.records
        if r["phase"] == phase and r["layer"] == layer and r["metric"] == metric
    ][-1]


class GradientMonitorTest(unittest.TestCase):
    def test_accumulated_gradients_and_shared_parameter(self):
        p = nn.Parameter(torch.tensor([1.0, 2.0]))
        missing = nn.Parameter(torch.tensor([1.0]))
        monitor = GradientMonitor(
            monitored_params={"layer.weight": p, "alias.weight": p, "unused.weight": missing},
            grad_interval=1,
        )
        (p.square().sum() / 2).backward()
        (p.square().sum() / 2).backward()
        monitor.capture_gradients(3)
        self.assertEqual(len(monitor.params), 2)
        self.assertAlmostEqual(value(monitor, "Gradient", "global", "norm"), math.sqrt(20))
        self.assertEqual(value(monitor, "Gradient", "global", "max_abs"), 4)
        self.assertEqual(value(monitor, "GradientParameter", "unused.weight", "present"), 0)
        self.assertEqual(value(monitor, "Gradient", "unused", "norm"), 0)
        torch.testing.assert_close(p.grad, torch.tensor([2.0, 4.0]))

    def test_zero_gradients_are_reported_and_nonfinite_is_not_hidden(self):
        p = nn.Parameter(torch.ones(2))
        monitor = GradientMonitor(monitored_params={"a.weight": p}, grad_interval=1)
        p.grad = torch.zeros(2)
        monitor.capture_gradients(0)
        self.assertEqual(value(monitor, "Gradient", "global", "norm"), 0)
        p.grad = torch.tensor([float("nan"), float("inf")])
        monitor.capture_gradients(1)
        self.assertTrue(math.isnan(value(monitor, "Gradient", "global", "norm")))
        self.assertEqual(value(monitor, "Gradient", "global", "nan_count"), 1)
        self.assertEqual(value(monitor, "Gradient", "global", "inf_count"), 1)

    def test_clipping_matches_actual_scale_and_frequency(self):
        p = nn.Parameter(torch.ones(2))
        monitor = GradientMonitor(monitored_params={"a.weight": p}, grad_interval=1)
        p.grad = torch.tensor([3.0, 4.0])
        monitor.capture_gradients(0)
        torch.nn.utils.clip_grad_norm_([p], 1.0)
        ratios = monitor.record_clipping(0)
        self.assertAlmostEqual(ratios["global"], 0.2, places=6)
        self.assertEqual(value(monitor, "Clipping", "global", "observed_frequency"), 1)
        p.grad = torch.zeros(2)
        monitor.capture_gradients(1)
        monitor.record_clipping(1)
        self.assertEqual(value(monitor, "Clipping", "global", "observed_frequency"), 0.5)
        self.assertIsNone(monitor.record_clipping(1))

    def test_actual_adamw_update_moments_and_denominator(self):
        model = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[1.0, -2.0]]))
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=0.03, betas=(0.8, 0.9), weight_decay=0.2, eps=1e-6
        )
        monitor = GradientMonitor(
            model,
            optimizer,
            grad_interval=1,
            param_interval=1,
            optimizer_interval=1,
            update_interval=1,
            adam_denominator=True,
        )
        model.weight.grad = torch.tensor([[0.5, -0.25]])
        before = model.weight.detach().clone()
        monitor.before_optimizer_step(0)
        optimizer.step()
        after = model.weight.detach().clone()
        monitor.after_optimizer_step(0)
        expected = (after - before).double().norm().item() / before.double().norm().item()
        self.assertAlmostEqual(value(monitor, "Update", "root", "weight_ratio"), expected, places=8)
        state = optimizer.state[model.weight]
        for key in ("exp_avg", "exp_avg_sq"):
            expected_rms = state[key].double().square().mean().sqrt().item()
            self.assertAlmostEqual(
                value(monitor, "Adam", "root", key + "_rms"), expected_rms, places=8
            )
        expected_denom = (state["exp_avg_sq"] / 0.1).sqrt() + 1e-6
        self.assertAlmostEqual(
            value(monitor, "Adam", "root", "bias_corrected_denominator_rms"),
            expected_denom.double().square().mean().sqrt().item(),
            places=7,
        )
        self.assertEqual(value(monitor, "Optimizer", "group_0", "lr"), 0.03)

    def test_missing_state_frozen_and_skipped_update(self):
        p = nn.Parameter(torch.tensor([2.0, 4.0]))
        frozen = nn.Parameter(torch.tensor([8.0]), requires_grad=False)
        optimizer = torch.optim.AdamW([p, frozen], lr=0.1)
        monitor = GradientMonitor(
            monitored_params={"a.weight": p, "frozen.weight": frozen},
            optimizer=optimizer,
            optimizer_interval=1,
            update_interval=1,
        )
        monitor.capture_optimizer_state(0)
        self.assertEqual(value(monitor, "AdamParameter", "a.weight", "exp_avg_available"), 0)
        self.assertTrue(math.isnan(value(monitor, "Adam", "a", "exp_avg_rms")))
        monitor.before_optimizer_step(0)
        monitor.after_optimizer_step(0, updated=False)
        self.assertEqual(value(monitor, "Optimizer", "global", "updated"), 0)
        self.assertEqual(value(monitor, "Update", "a", "weight_ratio"), 0)
        self.assertEqual(value(monitor, "Update", "frozen", "weight_ratio"), 0)
        self.assertFalse(optimizer.state)

    def test_snapshot_lifecycle_and_intervals(self):
        p = nn.Parameter(torch.tensor([1.0]))
        monitor = GradientMonitor(
            monitored_params={"a.weight": p}, grad_interval=3, update_interval=2
        )
        self.assertIsNone(monitor.capture_gradients(1))
        monitor.before_optimizer_step(2)
        with self.assertRaises(RuntimeError):
            monitor.before_optimizer_step(2)
        with self.assertRaises(ValueError):
            monitor.after_optimizer_step(3)
        monitor.after_optimizer_step(2, updated=False)
        monitor.before_optimizer_step(4)
        monitor.disable()
        self.assertFalse(monitor._snapshots)
        monitor.enable()
        monitor.before_optimizer_step(6)
        monitor.after_optimizer_step(6, updated=False)

    def test_filters_loss_and_csv_main_rank(self):
        with tempfile.TemporaryDirectory() as directory:
            model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 1))
            monitor = GradientMonitor(model, include=r"^0\.", exclude="bias", save_dir=directory)
            self.assertEqual(list(monitor.params), ["0.weight"])
            stats = monitor.record_loss(torch.tensor([1.0, 3.0]), 7)
            self.assertAlmostEqual(stats["mean"], 2, places=6)
            self.assertEqual(value(monitor, "Loss", "total", "max"), 3)
            monitor.flush()
            with (Path(directory) / "training_metrics.csv").open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(rows[0]["step"], "7")
            self.assertEqual(rows[0]["scope"], "rank_local")
            self.assertFalse(monitor.records)
            other = GradientMonitor(model, rank=1, save_dir=Path(directory) / "other")
            other.record_loss(torch.tensor(1.0), 0)
            other.flush()
            self.assertFalse(other.records)
            self.assertFalse((Path(directory) / "other").exists())

    def test_sharded_summary_uses_sum_and_max_not_world_average(self):
        p = nn.Parameter(torch.tensor([1.0, 2.0]))
        p.grad = torch.tensor([3.0, 4.0])
        group = object()
        monitor = GradientMonitor(
            monitored_params={"a.weight": p},
            sharded=True,
            shard_process_group=group,
            grad_interval=1,
        )
        operations = []

        def reduce(tensor, op, group):
            operations.append((op, group))
            if op == torch.distributed.ReduceOp.SUM:
                tensor.mul_(2)

        with (
            mock.patch("numerical_monitor.gradient_monitor.dist.is_initialized", return_value=True),
            mock.patch("numerical_monitor.gradient_monitor.dist.all_reduce", side_effect=reduce),
        ):
            monitor.capture_gradients(0)
        self.assertAlmostEqual(value(monitor, "Gradient", "global", "norm"), math.sqrt(50))
        self.assertEqual(value(monitor, "Gradient", "global", "max_abs"), 4)
        self.assertEqual(
            operations,
            [(torch.distributed.ReduceOp.SUM, group), (torch.distributed.ReduceOp.MAX, group)],
        )

    def test_unsampled_optimizer_steps_still_enforce_lifecycle(self):
        monitor = GradientMonitor(nn.Linear(1, 1), update_interval=100)
        with self.assertRaises(ValueError):
            monitor.after_optimizer_step(1)
        monitor.before_optimizer_step(1)
        with self.assertRaises(RuntimeError):
            monitor.before_optimizer_step(2)
        with self.assertRaises(ValueError):
            monitor.after_optimizer_step(2)
        monitor.after_optimizer_step(1, updated=False)

    def test_missing_double_shard_uses_identical_collective_dtype(self):
        p = nn.Parameter(torch.zeros(1, dtype=torch.float64))
        monitor = GradientMonitor(monitored_params={"w": p}, sharded=True)
        calls = []
        with (
            mock.patch.object(torch.distributed, "is_initialized", return_value=True),
            mock.patch.object(
                torch.distributed,
                "all_reduce",
                side_effect=lambda tensor, **kwargs: calls.append(tensor.dtype),
            ),
        ):
            monitor._summaries({"w": (None, p)})
            monitor._summaries({"w": (torch.ones_like(p), p)})
        self.assertEqual(calls, [torch.float64] * 4)

    def test_output_head_weight_norm_excludes_bias(self):
        model = nn.Sequential(nn.Linear(2, 1))
        with torch.no_grad():
            model[0].weight.copy_(torch.tensor([[3.0, 4.0]]))
            model[0].bias.fill_(12.0)
        monitor = GradientMonitor(model, output_heads=("0",), param_interval=1)
        monitor.before_optimizer_step(0)
        self.assertEqual(value(monitor, "Parameter", "0", "norm"), 13)
        self.assertEqual(value(monitor, "OutputHead", "0", "weight_norm"), 5)
        monitor.after_optimizer_step(0, updated=False)

    def test_double_precision_actual_update_is_not_rounded_away(self):
        p = nn.Parameter(torch.tensor([1.0], dtype=torch.float64))
        optimizer = torch.optim.SGD([p], lr=1e-10)
        monitor = GradientMonitor(
            monitored_params={"a.weight": p}, optimizer=optimizer, update_interval=1
        )
        p.grad = torch.ones_like(p)
        old = p.detach().clone()
        monitor.before_optimizer_step(0)
        optimizer.step()
        monitor.after_optimizer_step(0)
        expected = (p.detach() - old).norm().item() / (old.norm().item() + monitor.eps)
        actual = value(monitor, "Update", "a", "weight_ratio")
        self.assertGreater(actual, 0)
        self.assertAlmostEqual(actual / expected, 1, places=6)


if __name__ == "__main__":
    unittest.main()
