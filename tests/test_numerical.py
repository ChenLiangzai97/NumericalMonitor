"""Regression tests for forward statistics and the original public lifecycle."""

import csv
import math
import tempfile
import unittest

import torch
from torch import nn

from numerical_monitor.numerical_analyzer import NumericalAnalyzer
from numerical_monitor.tensor_filter import FilterRegistry


class NumericalAnalyzerTest(unittest.TestCase):
    def tearDown(self):
        FilterRegistry.clear()

    def test_hooks_never_change_model_values_and_flush_without_disk(self):
        model = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            model.weight.copy_(torch.eye(2))
        events = []
        monitor = NumericalAnalyzer(
            model, lambda *args: events.append(args), log_interval=1, save_dir=None
        )
        monitor.enable()
        monitor.begin_step(4)
        x = torch.tensor([[3.0, 4.0]], requires_grad=True)
        output = model(x)
        output.sum().backward()
        torch.testing.assert_close(output, x)
        torch.testing.assert_close(x.grad, torch.ones_like(x))
        monitor.step_forward(4)
        record = next(r for r in monitor.last_records if r["type"] == "Output")
        self.assertAlmostEqual(record["rms"], math.sqrt(12.5), places=5)
        self.assertEqual(record["max_abs"], 4)
        self.assertEqual(record["step"], 4)
        self.assertTrue(events)
        monitor.disable()
        model(x)
        self.assertEqual(monitor.detail_buffer, [])
        self.assertFalse(monitor.hook_handles)

    def test_batch_population_variance_and_parameter_not_split(self):
        model = nn.Linear(2, 2, bias=False)
        monitor = NumericalAnalyzer(model, log_interval=1, save_dir=None, batch_size=2)
        monitor.enable()
        model(torch.tensor([[0.0, 0.0], [10.0, 10.0]]))
        param = next(r for r in monitor.detail_buffer if r["type"] == "Param")
        self.assertFalse(param["batch_mode"])
        inputs = next(r for r in monitor.detail_buffer if r["type"] == "Input")
        self.assertTrue(inputs["batch_mode"])
        self.assertAlmostEqual(inputs["mean"], 5)
        self.assertAlmostEqual(inputs["std"], 5)
        self.assertEqual(inputs["max_diff"], 0)  # worst within-sample range
        monitor.disable()

    def test_nonfinite_batch_aggregation(self):
        monitor = NumericalAnalyzer(nn.Identity(), log_interval=1, save_dir=None, batch_size=2)
        monitor.enable()
        monitor.log_manual("test", "x", torch.tensor([[float("nan"), float("inf")], [0.0, 4.0]]))
        record = monitor.detail_buffer[-1]
        self.assertEqual(record["finite_count"], 2)
        self.assertEqual(record["nan_count"], 1)
        self.assertEqual(record["inf_count"], 1)
        self.assertAlmostEqual(record["mean"], 2)
        self.assertAlmostEqual(record["std"], 2)
        self.assertAlmostEqual(record["rms"], math.sqrt(8), places=5)
        monitor.disable()

    def test_sampling_resume_and_csv_step_identity(self):
        model = nn.Identity()
        with tempfile.TemporaryDirectory() as directory:
            monitor = NumericalAnalyzer(model, log_interval=10, save_dir=directory)
            monitor.enable()
            monitor.begin_step(21)
            model(torch.ones(2))
            self.assertEqual(monitor.detail_buffer, [])
            monitor.step_forward(21)
            monitor.begin_step(30)
            monitor.log_text("中文事件")
            model(torch.tensor([2.0, 8.0]))
            with self.assertRaises(ValueError):
                monitor.step_forward(31)
            monitor.step_forward(30)
            from pathlib import Path

            with (Path(directory) / "1/layer_stats_step_30.csv").open(
                encoding="utf-8-sig"
            ) as stream:
                rows = list(csv.DictReader(stream))
            self.assertTrue(rows)
            self.assertTrue(all(r["step"] == "30" for r in rows))
            monitor.disable()

    def test_filter_recursive_outputs_and_explicit_roles(self):
        class Structured(nn.Module):
            def forward(self, x):
                return {"logits": x * 2, "gt": [x * 100]}

        model = Structured()
        FilterRegistry.register(lambda pos, tensor, mod, identity: "gt" in pos)
        monitor = NumericalAnalyzer(
            model,
            log_interval=1,
            save_dir=None,
            output_head_modules=[r"model.root$"],
            residual_modules=[r"model.root$"],
        )
        monitor.enable()
        model(torch.tensor([1.0, 3.0]))
        outputs = [r for r in monitor.detail_buffer if r["type"] == "Output"]
        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0]["module"], "root.out_logits")
        self.assertEqual(outputs[0]["role"], "residual_stream,output_head_logit")
        monitor.disable()

    def test_non_main_rank_and_independent_instances(self):
        model = nn.Identity()
        first = NumericalAnalyzer(model, save_dir=None)
        second = NumericalAnalyzer(model, save_dir=None, rank=1)
        self.assertIsNot(first, second)
        second.enable()
        model(torch.ones(1))
        self.assertEqual(second.detail_buffer, [])
        self.assertFalse(second.hook_handles)

    def test_checkpoint_recomputation_deduplication_is_explicit(self):
        from torch.utils.checkpoint import checkpoint

        for deduplicate, expected in ((False, 2), (True, 1)):
            model = nn.Linear(2, 2)
            monitor = NumericalAnalyzer(
                model, log_interval=1, save_dir=None, deduplicate_calls=deduplicate
            )
            monitor.enable()
            monitor.begin_step(12)
            x = torch.ones(1, 2, requires_grad=True)
            checkpoint(model, x, use_reentrant=True).sum().backward()
            outputs = [r for r in monitor.detail_buffer if r["type"] == "Output"]
            self.assertEqual(len(outputs), expected)
            self.assertTrue(all(r["step"] == 12 for r in outputs))
            # Parameters retain a single record across recomputation.
            self.assertEqual(len([r for r in monitor.detail_buffer if r["type"] == "Param"]), 2)
            monitor.disable()

    def test_shared_parameters_counted_once(self):
        class Shared(nn.Module):
            def __init__(self):
                super().__init__()
                self.a = nn.Linear(2, 2, bias=False)
                self.b = nn.Linear(2, 2, bias=False)
                self.b.weight = self.a.weight

            def forward(self, x):
                return self.b(self.a(x))

        model = Shared()
        monitor = NumericalAnalyzer(model, log_interval=1, save_dir=None)
        monitor.enable()
        model(torch.ones(1, 2))
        self.assertEqual(len([r for r in monitor.detail_buffer if r["type"] == "Param"]), 1)
        monitor.disable()


if __name__ == "__main__":
    unittest.main()
