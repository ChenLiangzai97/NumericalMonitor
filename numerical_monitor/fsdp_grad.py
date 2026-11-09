"""FSDP1 compatibility collectors derived from the supplied shard monitors.

All ranks must call the same methods. ``process_group`` must contain each shard
exactly once (the shard group, never the HSDP replica group). No full parameters
or gradients are gathered. Call the fallback after accumulation, synchronization
and AMP unscale, before clipping. The optional private hook has a narrower
single-backward contract and is never installed merely by importing this module.
"""

import inspect
import math

import torch
import torch.distributed as dist


class _Work:
    """Own snapshots until all asynchronous reductions have completed."""

    def __init__(self, square, maximum=None, counts=None, works=()):
        self.square = square
        self.maximum = maximum
        self.counts = counts
        self.works = works
        self.done = False

    def wait(self):
        if not self.done:
            for work in self.works:
                if work is not None:
                    work.wait()
            self.done = True
        return self


class _AsyncNormResult:
    def __init__(self, square, work):
        self._square = square
        self._work = work
        self._result = None

    def wait(self):
        if self._result is None:
            self._work.wait()
            self._result = self._square.sqrt()
        return self._result


def _reduce(tensor, op, group, async_op=True):
    if dist.is_available() and dist.is_initialized() and tensor.numel():
        return dist.all_reduce(tensor, op=op, group=group, async_op=async_op)
    return None


@torch.no_grad()
def reduce_shard_grad_norms(sq_norms, process_group=None, is_squared=True, async_op=False):
    """Reduce local squares, or square local norms first when is_squared=False.

    Unlike the old helper, this does not mutate the caller's tensor.
    """
    square = sq_norms.detach().clone() if is_squared else sq_norms.detach().square()
    work = _Work(square, works=[_reduce(square, dist.ReduceOp.SUM, process_group)])
    result = _AsyncNormResult(square, work)
    return result if async_op else result.wait()


class FSDPGradNormOptimizer:
    """Asynchronous per-parameter shard statistics, preserving prepare/finalize.

    Shared parameter aliases are deduplicated by identity; their group membership
    is merged. ``last_stats`` is available on every rank; only the writer invokes
    logging callbacks. Zero gradients and missing gradients are retained.
    """

    def __init__(
        self,
        monitored_params,
        param_to_groups,
        groups_info,
        top_k=10,
        process_group=None,
        is_main_process=True,
    ):
        if top_k < 0:
            raise ValueError("top_k must be nonnegative")
        self._param_names = []
        self._param_refs = []
        self._memberships = []
        canonical = {}
        for name, param in monitored_params.items():
            if id(param) in canonical:
                self._memberships[canonical[id(param)]].update(param_to_groups.get(name, ()))
                continue
            canonical[id(param)] = len(self._param_refs)
            self._param_names.append(name)
            self._param_refs.append(param)
            self._memberships.append(set(param_to_groups.get(name, ())))
        self._num_params = len(self._param_refs)
        devices = {p.device for p in self._param_refs}
        if len(devices) > 1:
            raise ValueError("One collector requires parameters on one device")
        self._device = self._param_refs[0].device if self._param_refs else torch.device("cpu")
        self._process_group = process_group
        self._is_main_process = is_main_process
        self._groups_info = dict(groups_info)
        self._k = min(top_k, self._num_params)
        self._pending = None
        self.last_stats = None

    @staticmethod
    def _measure(grad):
        grad = grad.detach()
        if grad.is_sparse:
            grad = grad.coalesce().values()
        if hasattr(grad, "to_local"):
            grad = grad.to_local()
        grad = grad.float()
        if not grad.numel():
            return (
                grad.new_zeros(()),
                grad.new_zeros(()),
                torch.zeros(2, device=grad.device, dtype=torch.int64),
            )
        return (
            grad.square().sum(),
            grad.abs().amax(),
            torch.stack([grad.isnan().sum(), grad.isinf().sum()]),
        )

    def _empty(self):
        return _Work(
            torch.zeros(self._num_params, device=self._device),
            torch.zeros(self._num_params, device=self._device),
            torch.zeros((2, self._num_params), dtype=torch.int64, device=self._device),
        )

    def _launch(self, result):
        result.works = [
            _reduce(result.square, dist.ReduceOp.SUM, self._process_group),
            _reduce(result.maximum, dist.ReduceOp.MAX, self._process_group),
            _reduce(result.counts, dist.ReduceOp.SUM, self._process_group),
        ]
        self._pending = result
        return result

    @torch.no_grad()
    def prepare(self):
        """Snapshot final unscaled gradients before optimizer mutates them."""
        if self._pending is not None:
            raise RuntimeError("finalize the previous prepare before preparing again")
        result = self._empty()
        for i, param in enumerate(self._param_refs):
            if param.grad is not None:
                square, maximum, counts = self._measure(param.grad)
                result.square[i] = square
                result.maximum[i] = maximum
                result.counts[:, i] = counts
        return self._launch(result)

    @torch.no_grad()
    def finalize(self, work, cur_iter_num, tb_scalar_func=None, tb_text_func=None, logger=None):
        if work is None:
            return None
        if work is not self._pending:
            raise ValueError("work does not belong to the pending prepare")
        work.wait()
        self._pending = None
        # One device-to-host transfer for floating statistics and one for counts.
        values = torch.stack([work.square, work.maximum]).cpu().tolist()
        counts = work.counts.cpu().tolist()
        rows = {}
        for i, name in enumerate(self._param_names):
            rows[name] = dict(
                norm=math.sqrt(values[0][i]),
                max_abs=values[1][i],
                nan_count=counts[0][i],
                inf_count=counts[1][i],
            )

        def aggregate(indices):
            return dict(
                norm=math.sqrt(sum(values[0][i] for i in indices)),
                max_abs=(
                    float("nan")
                    if any(counts[0][i] for i in indices)
                    else max((values[1][i] for i in indices), default=0.0)
                ),
                nan_count=sum(counts[0][i] for i in indices),
                inf_count=sum(counts[1][i] for i in indices),
            )

        groups = {
            name: aggregate(
                [i for i, membership in enumerate(self._memberships) if name in membership]
            )
            for name in self._groups_info
        }
        self.last_stats = dict(
            parameters=rows, groups=groups, global_stats=aggregate(range(self._num_params))
        )
        if not self._is_main_process:
            return self.last_stats
        scalar = tb_scalar_func or (lambda *args: None)
        text = tb_text_func or (lambda *args: None)
        total = self.last_stats["global_stats"]
        scalar("global_grad_norm/total_grad_norm", total["norm"], cur_iter_num)
        average = sum(row["norm"] for row in rows.values()) / max(len(rows), 1)
        scalar("global_grad_norm/avg_grad_norm", average, cur_iter_num)
        text("global_grad_norm/total_grad_norm", str(total["norm"]), cur_iter_num)
        text("global_grad_norm/avg_grad_norm", str(average), cur_iter_num)
        for key in ("max_abs", "nan_count", "inf_count"):
            scalar(f"global_grad_norm/{key}", total[key], cur_iter_num)
        for name, row in rows.items():
            for key, value in row.items():
                scalar(f"parameter_grad/{name}/{key}", value, cur_iter_num)
        ordered = sorted(
            rows,
            key=lambda name: (math.isnan(rows[name]["norm"]), rows[name]["norm"]),
            reverse=True,
        )
        for rank, name in enumerate(ordered[: self._k], 1):
            norm = rows[name]["norm"]
            scalar(f"global_grad_norm/top_{rank}_max_grad", norm, cur_iter_num)
            text(f"global_grad_norm/top_{rank}_param", f"{name}: {norm}", cur_iter_num)
            if logger is not None:
                logger.info("top%d_param %s: %s %s", rank, name, norm, cur_iter_num)
        for name, row in groups.items():
            loss_group = self._groups_info[name]
            prefix = f"{name}_" if name else ""
            for key, value in row.items():
                suffix = "total_grad_norm" if key == "norm" else key
                scalar(f"{loss_group}/{prefix}{suffix}", value, cur_iter_num)
            indices = [i for i, m in enumerate(self._memberships) if name in m]
            group_average = sum(rows[self._param_names[i]]["norm"] for i in indices) / max(
                len(indices), 1
            )
            scalar(f"{loss_group}/{prefix}avg_grad_norm", group_average, cur_iter_num)
            category = name if name else loss_group
            text(f"{category}/total_grad_norm", str(row["norm"]), cur_iter_num)
            text(f"{category}/avg_grad_norm", str(group_average), cur_iter_num)
            if indices:
                top_name = max(
                    (self._param_names[i] for i in indices),
                    key=lambda name: (math.isnan(rows[name]["norm"]), rows[name]["norm"]),
                )
                scalar(f"{loss_group}/{prefix}top_1_max_grad", rows[top_name]["norm"], cur_iter_num)
                text(
                    f"{category}/top1_param", f"{top_name}: {rows[top_name]['norm']}", cur_iter_num
                )
        return self.last_stats


class HSDPGradStatsHook(FSDPGradNormOptimizer):
    """Opt-in compatibility hook using private FSDP1 callback metadata.

    Pass ``enable_private_hook=True`` explicitly. Only a single synchronized
    backward per optimizer step is supported. For gradient accumulation use
    FSDPGradNormOptimizer.prepare after the final backward instead. ``grad_scale``
    rescales finite magnitudes only; overflow counts describe the captured shard.
    Private callback signatures and metadata are checked before patch installation.
    GPU stream/per-version validation is still required on the deployment stack.
    """

    def __init__(
        self,
        monitored_params,
        param_to_groups,
        groups_info,
        top_k,
        process_group,
        is_main_process,
        fsdp_modules,
        *,
        enable_private_hook=False,
    ):
        if not enable_private_hook:
            raise RuntimeError(
                "Private FSDP1 hook is opt-in; use enable_private_hook=True or the prepare/finalize collector"
            )
        super().__init__(
            monitored_params, param_to_groups, groups_info, top_k, process_group, is_main_process
        )
        self._installed = False
        self._active = False
        self._captured = self._empty()
        self._pending_step = None
        self._grad_scale = 1.0
        self._handle_meta = {}
        ids = {id(p): i for i, p in enumerate(self._param_refs)}
        visited = set()
        for root in fsdp_modules:
            for module in root.modules():
                handles = [getattr(module, "_handle", None)]
                handles += list(getattr(module, "_all_handles", ()) or ())
                for handle in handles:
                    if handle is None or id(handle) in visited:
                        continue
                    visited.add(id(handle))
                    flat = handle.flat_param
                    infos = getattr(flat, "_shard_param_infos", None)
                    params = getattr(flat, "_params", None)
                    if infos is None or params is None:
                        raise RuntimeError(
                            "Unsupported FSDP1 metadata; use use_orig_params=True and prepare/finalize"
                        )
                    segments = []
                    for param, info in zip(params, infos):
                        pid = ids.get(id(param))
                        if pid is not None and getattr(info, "in_shard", False):
                            segments.append((pid, info.offset_in_shard, info.numel_in_shard))
                    self._handle_meta[id(handle)] = segments
        if not self._handle_meta:
            raise RuntimeError("No compatible FSDP1 handles found; no patch was installed")
        import torch.distributed.fsdp._runtime_utils as runtime

        callback = getattr(runtime, "_post_reduce_grad_callback", None)
        if callback is None or tuple(inspect.signature(callback).parameters) != (
            "state",
            "handle",
            "grad_to_offload",
        ):
            raise RuntimeError("Unsupported FSDP1 callback signature; no patch was installed")
        if getattr(callback, "_numerical_monitor_patch", False):
            raise RuntimeError("Another numerical monitor owns the FSDP1 callback")
        self._runtime = runtime
        self._original = callback

        def patched(state, handle, grad_to_offload):
            if (
                self._active
                and getattr(state, "_sync_gradients", True)
                and grad_to_offload is not None
            ):
                self._capture(handle, grad_to_offload)
            return callback(state, handle, grad_to_offload)

        patched._numerical_monitor_patch = True
        self._patched = patched
        runtime._post_reduce_grad_callback = patched
        self._installed = True

    @torch.no_grad()
    def _capture(self, handle, grad):
        for pid, start, length in self._handle_meta.get(id(handle), ()):
            square, maximum, counts = self._measure(grad[start : start + length])
            # Replace the latest shard, rather than summing norms from repeated
            # callbacks: squared norms of accumulated gradients are not additive.
            self._captured.square[pid] = square / self._grad_scale**2
            self._captured.maximum[pid] = maximum / self._grad_scale
            self._captured.counts[:, pid] = counts

    def reset(self, *, grad_scale=1.0, gradient_accumulation=False):
        if self._pending is not None:
            raise RuntimeError("Finalize pending statistics before reset")
        if gradient_accumulation:
            raise ValueError("Private hook does not support accumulation; use prepare/finalize")
        if not math.isfinite(grad_scale) or grad_scale <= 0:
            raise ValueError("grad_scale must be finite and positive")
        self._grad_scale = grad_scale
        self._captured = self._empty()
        self._active = True

    def launch_all_reduce(self, step):
        if not self._active or self._pending is not None:
            raise RuntimeError("reset once before backward and finalize before next launch")
        self._active = False
        self._pending_step = step
        return self._launch(self._captured)

    def try_finalize(self, step, tb_scalar_func=None, tb_text_func=None, logger=None):
        if self._pending is None or self._pending_step != step:
            return False
        self.finalize(self._pending, step, tb_scalar_func, tb_text_func, logger)
        self._pending_step = None
        return True

    def advance_buffer(self):
        """Compatibility no-op: pending snapshots own their storage."""

    def uninstall_patch(self):
        self._active = False
        try:
            if self._pending is not None:
                self._pending.wait()
        finally:
            self._pending = None
            if self._installed:
                if self._runtime._post_reduce_grad_callback is self._patched:
                    self._runtime._post_reduce_grad_callback = self._original
                self._installed = False

    close = uninstall_patch

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.uninstall_patch()


def _extract_shard_pg_from_fsdp1(wrapper):
    module = getattr(wrapper, "module", wrapper)
    return getattr(module, "process_group", None)


def _find_fsdp_root_modules(trainer):
    from torch.distributed.fsdp import FullyShardedDataParallel

    wrappers = getattr(trainer.model, "wrappers", ())
    return [
        module
        for wrapper in wrappers
        if isinstance(module := getattr(wrapper, "module", wrapper), FullyShardedDataParallel)
    ]


def extract_fsdp1_config(trainer):
    """Extract legacy trainer configuration, rejecting ambiguous shard groups."""
    if getattr(trainer.args, "parallel_strategy", "ddp") != "fsdp1":
        return dict(
            is_fsdp1=False, process_group=None, needs_shard_reduction=False, fsdp_modules=[]
        )
    wrappers = list(getattr(trainer.model, "wrappers", ()))
    groups = [_extract_shard_pg_from_fsdp1(w) for w in wrappers]
    distinct = {id(group): group for group in groups if group is not None}
    if len(distinct) > 1:
        raise ValueError("Multiple FSDP shard groups require separate collectors")
    return dict(
        is_fsdp1=True,
        process_group=next(iter(distinct.values()), None),
        needs_shard_reduction=not any(getattr(w, "delay_grad_all_reduce", False) for w in wrappers),
        fsdp_modules=_find_fsdp_root_modules(trainer),
    )
