"""原有 FSDP 采集接口的回归测试，覆盖分片归约及异步生命周期。"""

import math
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from numerical_monitor.fsdp_grad import (
    FSDPGradNormOptimizer,
    HSDPGradStatsHook,
    reduce_shard_grad_norms,
)


def _distributed_worker(rank, rendezvous, output):
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    try:
        p = torch.nn.Parameter(torch.zeros(2))
        p.grad = torch.tensor([3.0, 0.0] if rank == 0 else [4.0, 12.0])
        monitor = FSDPGradNormOptimizer(
            {"w": p}, {"w": {"layer"}}, {"layer": "train"}, 1, dist.group.WORLD, rank == 0
        )
        result = monitor.finalize(monitor.prepare(), 7)
        assert result["global_stats"]["norm"] == 13.0
        assert result["groups"]["layer"]["max_abs"] == 12.0
        p.grad = torch.tensor([float("nan"), float("inf")]) if rank == 0 else torch.zeros(2)
        result = monitor.finalize(monitor.prepare(), 8)
        assert result["global_stats"]["nan_count"] == 1
        assert result["global_stats"]["inf_count"] == 1
        Path(output, str(rank)).write_text("passed")
    finally:
        dist.destroy_process_group()


class FSDPCollectorTest(unittest.TestCase):
    def test_norm_max_counts_shared_parameters_and_zero(self):
        p = torch.nn.Parameter(torch.zeros(2))
        p.grad = torch.tensor([3.0, 4.0])
        zero = torch.nn.Parameter(torch.zeros(1))
        zero.grad = torch.zeros(1)
        monitor = FSDPGradNormOptimizer(
            {"w": p, "alias": p, "zero": zero},
            {"w": {"a"}, "alias": {"b"}, "zero": {"a"}},
            {"a": "train", "b": "train"},
            3,
            None,
            True,
        )
        scalars = {}
        result = monitor.finalize(
            monitor.prepare(), 1, lambda tag, value, step: scalars.update({tag: value})
        )
        self.assertEqual(result["global_stats"]["norm"], 5)
        self.assertEqual(len(result["parameters"]), 2)
        self.assertEqual(result["groups"]["b"]["norm"], 5)
        self.assertEqual(result["groups"]["a"]["max_abs"], 4)
        self.assertEqual(scalars["global_grad_norm/avg_grad_norm"], 2.5)
        p.grad.zero_()
        result = monitor.finalize(monitor.prepare(), 2)
        self.assertEqual(result["global_stats"]["norm"], 0)

    def test_pending_snapshot_survives_mutation_and_blocks_reuse(self):
        p = torch.nn.Parameter(torch.zeros(2))
        p.grad = torch.tensor([3.0, 4.0])
        monitor = FSDPGradNormOptimizer({"w": p}, {}, {})
        pending = monitor.prepare()
        with self.assertRaises(RuntimeError):
            monitor.prepare()
        p.grad.zero_()
        self.assertEqual(monitor.finalize(pending, 0)["global_stats"]["norm"], 5)

    def test_nonfinite_and_empty_parameters(self):
        p = torch.nn.Parameter(torch.zeros(3))
        p.grad = torch.tensor([float("nan"), float("inf"), -float("inf")])
        monitor = FSDPGradNormOptimizer({"w": p}, {}, {})
        stats = monitor.finalize(monitor.prepare(), 0)["global_stats"]
        self.assertTrue(math.isnan(stats["norm"]))
        self.assertEqual(stats["nan_count"], 1)
        self.assertEqual(stats["inf_count"], 2)
        empty = FSDPGradNormOptimizer({}, {}, {})
        self.assertEqual(empty.finalize(empty.prepare(), 0)["global_stats"]["norm"], 0)

    def test_small_buffers_use_sum_max_sum_on_shard_group(self):
        p = torch.nn.Parameter(torch.zeros(2))
        p.grad = torch.tensor([3.0, 0.0])
        group = object()
        monitor = FSDPGradNormOptimizer({"w": p}, {}, {}, process_group=group)
        calls = []

        def reduce(tensor, op, group, async_op):
            calls.append((op, group, async_op))
            if len(calls) == 1:
                tensor.add_(160.0)  # Remote shard [4, 12].
            elif op == dist.ReduceOp.MAX:
                tensor.clamp_(min=12.0)
            return mock.Mock()

        with (
            mock.patch.object(dist, "is_initialized", return_value=True),
            mock.patch.object(dist, "all_reduce", side_effect=reduce),
        ):
            result = monitor.finalize(monitor.prepare(), 1)
        self.assertEqual(result["global_stats"]["norm"], 13.0)
        self.assertEqual(result["global_stats"]["max_abs"], 12.0)
        self.assertEqual(
            [c[0] for c in calls], [dist.ReduceOp.SUM, dist.ReduceOp.MAX, dist.ReduceOp.SUM]
        )
        self.assertTrue(all(c[1] is group and c[2] for c in calls))

    def test_nonwriter_does_not_log(self):
        p = torch.nn.Parameter(torch.zeros(1))
        monitor = FSDPGradNormOptimizer({"w": p}, {}, {}, is_main_process=False)
        fail = mock.Mock(side_effect=AssertionError("unexpected logging"))
        monitor.finalize(monitor.prepare(), 0, fail, fail)
        fail.assert_not_called()

    def test_reduce_helper_honors_is_squared_and_does_not_mutate(self):
        values = torch.tensor([3.0, 4.0])
        result = reduce_shard_grad_norms(values, is_squared=False, async_op=True)
        self.assertTrue(torch.equal(result.wait(), values))
        self.assertTrue(torch.equal(values, torch.tensor([3.0, 4.0])))
        self.assertTrue(torch.equal(reduce_shard_grad_norms(torch.tensor([9.0, 16.0])), values))

    def test_private_hook_requires_opt_in_and_checks_before_installing(self):
        import torch.distributed.fsdp._runtime_utils as runtime

        original = runtime._post_reduce_grad_callback
        with self.assertRaisesRegex(RuntimeError, "opt-in"):
            HSDPGradStatsHook({}, {}, {}, 1, None, True, [])
        with self.assertRaisesRegex(RuntimeError, "No compatible"):
            HSDPGradStatsHook({}, {}, {}, 1, None, True, [], enable_private_hook=True)
        self.assertIs(runtime._post_reduce_grad_callback, original)

    def test_private_hook_capture_lifecycle_scaling_and_restore(self):
        import torch.distributed.fsdp._runtime_utils as runtime

        p = torch.nn.Parameter(torch.zeros(2))
        flat = SimpleNamespace(
            _params=[p],
            _shard_param_infos=[
                SimpleNamespace(in_shard=True, offset_in_shard=0, numel_in_shard=2)
            ],
        )
        handle = SimpleNamespace(flat_param=flat)
        root = torch.nn.Module()
        root._handle = handle

        def original(state, handle, grad_to_offload):
            return "original called"

        with mock.patch.object(runtime, "_post_reduce_grad_callback", original):
            hook = HSDPGradStatsHook(
                {"w": p},
                {"w": {"layer"}},
                {"layer": "train"},
                1,
                None,
                True,
                [root],
                enable_private_hook=True,
            )
            with self.assertRaises(ValueError):
                hook.reset(gradient_accumulation=True)
            hook.reset(grad_scale=2)
            self.assertEqual(
                runtime._post_reduce_grad_callback(
                    SimpleNamespace(_sync_gradients=True), handle, torch.tensor([6.0, 8.0])
                ),
                "original called",
            )
            hook.launch_all_reduce(9)
            with self.assertRaises(RuntimeError):
                hook.reset()
            self.assertFalse(hook.try_finalize(8))
            self.assertTrue(hook.try_finalize(9))
            self.assertEqual(hook.last_stats["global_stats"]["norm"], 5)
            self.assertEqual(hook.last_stats["global_stats"]["max_abs"], 4)
            hook.advance_buffer()
            hook.uninstall_patch()
            self.assertIs(runtime._post_reduce_grad_callback, original)

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "Gloo unavailable")
    def test_two_rank_shard_sum_max_and_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            # 部分执行环境禁止 Gloo 使用的本地 TCP socket；先探测再运行双进程测试。
            try:
                dist.init_process_group(
                    "gloo", init_method=f"file://{directory}/probe", rank=0, world_size=1
                )
            except RuntimeError as error:
                if "Operation not permitted" in str(error):
                    self.skipTest("Executor forbids Gloo local TCP sockets")
                raise
            else:
                dist.destroy_process_group()
            mp.spawn(
                _distributed_worker,
                args=(os.path.join(directory, "rendezvous"), directory),
                nprocs=2,
                join=True,
            )
            self.assertEqual(Path(directory, "0").read_text(), "passed")
            self.assertEqual(Path(directory, "1").read_text(), "passed")


if __name__ == "__main__":
    unittest.main()
