"""Forward/parameter monitoring derived from the supplied NumericalAnalyzer.

Forward records describe this rank's batch, not the distributed training batch.
This module never issues collectives, changes model outputs, or keeps graphs.
FSDP parameters observed while sharded are explicitly labeled local_shard.
"""

import csv
import inspect
import math
from pathlib import Path
import re

import torch
from torch import nn

from .metrics import tensor_metrics
from .tensor_filter import FilterRegistry


class MetricKey:
    MAX = "max"
    MIN = "min"
    MAX_DIFF = "max_diff"
    MEAN = "mean"
    STD = "std"
    ZERO_RATIO = "zero_ratio"
    RUNNING_MEAN = "running_mean"
    RUNNING_VAR = "running_var"
    RMS = "rms"
    MAX_ABS = "max_abs"
    L2_NORM = "l2_norm"
    NAN_COUNT = "nan_count"
    INF_COUNT = "inf_count"


def _patterns(patterns):
    if isinstance(patterns, str):
        patterns = [patterns]
    return [re.compile(p) for p in (patterns or ())]


def _matches(patterns, name):
    return any(p.search(name) for p in patterns)


class NumericalAnalyzer:
    """Sample module inputs/outputs and parameters without modifying execution.

    Original constructor arguments and step()/step_forward() remain available.
    Instances are intentionally independent, allowing train/eval isolation.

    Args:
        container: {'net_parts': {'name': module}}, or a plain nn.Module.
        log_scalar_fn: TensorBoard-compatible (tag, value, step) callable.
        groups: Optional net_parts keys to monitor.
        log_interval: Sample one optimizer step every log_interval steps.
        save_dir: CSV directory; None disables disk output, not scalar logging.
        rank: Global rank; -1 means infer from initialized distributed state.
        batch_size: Enables per-sample ranges for activation tensors only.
        is_fsdp2: Compatibility option; sharded values are never called global.
        include_modules: Regexes matched against group.module; None means all.
        exclude_modules: Regexes excluded from hook registration.
        residual_modules: Explicit modules whose outputs are residual streams.
        output_head_modules: Explicit modules whose outputs are head logits.
        scalar_metrics: Metric names exported to the scalar logger.
        batch_dim: Explicit activation batch dimension; None disables splitting.
        scan_parameters: Disable if parameters are collected by GradientMonitor.
        deduplicate_calls: Keep first call per module per step; suppresses
            checkpoint recomputation but also repeated/shared module calls.
    """

    def __init__(
        self,
        container=None,
        log_scalar_fn=None,
        groups=None,
        log_interval=1000,
        save_dir="./numerical_logs",
        rank=-1,
        batch_size=0,
        is_fsdp2=False,
        *,
        include_modules=None,
        exclude_modules=None,
        residual_modules=None,
        output_head_modules=None,
        scalar_metrics=None,
        batch_dim=0,
        scan_parameters=True,
        deduplicate_calls=False,
    ):
        if log_interval < 1:
            raise ValueError("log_interval must be positive")
        if batch_size < 0:
            raise ValueError("batch_size must be nonnegative")
        if rank == -1 and torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
        self.rank = rank
        self.is_main_rank = rank in (-1, 0)
        self._is_fsdp2 = is_fsdp2
        self.is_enable = False
        self.log_interval = log_interval
        self.step_counter = 0
        self.batch_size = batch_size
        self.batch_dim = batch_dim
        self.scan_parameters = scan_parameters
        self.deduplicate_calls = deduplicate_calls
        self.scalar_metrics = (
            tuple(scalar_metrics)
            if scalar_metrics is not None
            else ("rms", "max_abs", "l2_norm", "nan_count", "inf_count")
        )
        self.log_scalar_fn = log_scalar_fn
        self.base_save_dir = save_dir
        self.active_save_dir = None
        self.case_counter = 0
        self.hook_handles = []
        self.pre_hook_handles = []
        self.detail_buffer = []
        self.text_buffer = []
        self.last_records = []
        self.module_mapping = {}
        self.monitored_modules = {}
        self._scanned_param_modules = set()
        self._scanned_parameters = set()
        self._seen_calls = set()
        self._has_scanned_params_this_step = False
        self._include = _patterns(include_modules)
        self._exclude = _patterns(exclude_modules)
        self._residual = _patterns(residual_modules)
        self._heads = _patterns(output_head_modules)
        if isinstance(container, nn.Module):
            container = {"net_parts": {"model": container}}
        parts = (container or {}).get("net_parts", {})
        for name in groups if groups is not None else parts:
            module = parts.get(name)
            if isinstance(module, nn.Module):
                self.monitored_modules[name] = module
        self._build_mapping()

    def _build_mapping(self):
        for group_name, root in self.monitored_modules.items():
            for name, module in root.named_modules():
                layer = name or "root"
                full_name = f"{group_name}.{layer}"
                if self._include and not _matches(self._include, full_name):
                    continue
                if _matches(self._exclude, full_name):
                    continue
                # Shared modules get one canonical identity, just as parameters do.
                self.module_mapping.setdefault(
                    module,
                    {
                        "group": group_name,
                        "module": layer,
                        "full_name": full_name,
                        "source_loc": None,
                    },
                )

    def enable(self):
        if not self.is_main_rank or self.is_enable:
            return
        self.is_enable = True
        self.case_counter += 1
        self.active_save_dir = (
            str(Path(self.base_save_dir) / str(self.case_counter)) if self.base_save_dir else None
        )
        self._register_hooks()

    def disable(self):
        for handle in self.hook_handles + self.pre_hook_handles:
            handle.remove()
        self.hook_handles.clear()
        self.pre_hook_handles.clear()
        self.is_enable = False
        self.detail_buffer.clear()
        self.text_buffer.clear()
        self._reset_step_state()

    close = disable

    def _reset_step_state(self):
        self._scanned_param_modules.clear()
        self._scanned_parameters.clear()
        self._seen_calls.clear()
        self._has_scanned_params_this_step = False

    def begin_step(self, step):
        """Set the optimizer step BEFORE forward; call once per accumulated step."""
        if step < 0:
            raise ValueError("step must be nonnegative")
        if self.detail_buffer:
            raise RuntimeError("flush the previous step with step_forward() first")
        self.step_counter = int(step)
        self._reset_step_state()

    def is_log_step(self, step=None):
        current = self.step_counter if step is None else step
        return self.is_main_rank and self.is_enable and current % self.log_interval == 0

    @staticmethod
    def _source_location():
        for frame in inspect.stack()[2:]:
            if frame.filename != __file__ and "torch/nn/modules" not in frame.filename:
                return f"{Path(frame.filename).name}:{frame.lineno}"
        return "Unknown"

    def log_text(self, text, step=None):
        if self.is_enable and self.is_main_rank:
            self.text_buffer.append(
                {
                    "step": self.step_counter if step is None else step,
                    "source_loc": self._source_location(),
                    "text": text,
                }
            )

    def log_manual(self, group_name, module_name, tensor):
        if self.is_log_step() and isinstance(tensor, torch.Tensor):
            self._compute_and_buffer(
                tensor, group_name, module_name, "Manual", source_loc=self._source_location()
            )

    def _register_hooks(self):
        if self.hook_handles:
            return
        for module in self.module_mapping:
            # These hooks return None and never materialize DTensors collectively.
            self.pre_hook_handles.append(module.register_forward_pre_hook(self._pre_forward_hook))
            self.hook_handles.append(module.register_forward_hook(self._forward_hook))

    def _pre_forward_hook(self, module, args):
        if not self.is_log_step() or not self.scan_parameters:
            return
        if id(module) not in self._scanned_param_modules:
            self._scanned_param_modules.add(id(module))
            self._scan_module_params_and_buffers(module)

    def _scan_module_params_and_buffers(self, module):
        identity = self.module_mapping.get(module)
        if identity is None:
            return
        for name, param in module.named_parameters(recurse=False):
            if id(param) in self._scanned_parameters:
                continue
            self._scanned_parameters.add(id(param))
            old_len = len(self.detail_buffer)
            self._compute_and_buffer(
                param,
                identity["group"],
                f"{identity['module']}.{name}",
                "Param",
                source_loc=identity.get("source_loc") or "Static",
            )
            if (
                name == "weight"
                and len(self.detail_buffer) > old_len
                and _matches(self._heads, identity["full_name"])
            ):
                self.detail_buffer[-1]["role"] = "output_head_weight"
        for name in ("running_mean", "running_var"):
            value = getattr(module, name, None)
            if isinstance(value, torch.Tensor):
                self._compute_and_buffer(
                    value,
                    identity["group"],
                    f"{identity['module']}.{name}",
                    "Buffer",
                    is_running_stat=True,
                )

    def _scan_parameters_and_buffers(self):
        """Compatibility method; only scans local values, never gathers FSDP."""
        for module in self.module_mapping:
            self._scan_module_params_and_buffers(module)

    def _forward_hook(self, module, inputs, output):
        if not self.is_log_step():
            return
        if self.deduplicate_calls and id(module) in self._seen_calls:
            return
        self._seen_calls.add(id(module))
        identity = self.module_mapping[module]
        if identity["source_loc"] is None:
            identity["source_loc"] = self._source_location()
        chain = FilterRegistry.get_chain(identity["group"])
        for kind, value, prefix in (
            ("Input", inputs, "in"),
            ("Output", output, "out" if isinstance(output, (tuple, list, dict)) else ""),
        ):
            for position, tensor in self._extract_tensors_with_positions(value, prefix):
                if chain.should_skip(position, tensor, module, identity):
                    continue
                name = identity["module"] + (f".{position}" if position else "")
                old_len = len(self.detail_buffer)
                self._compute_and_buffer(
                    tensor, identity["group"], name, kind, source_loc=identity["source_loc"]
                )
                if len(self.detail_buffer) == old_len:
                    continue
                if kind == "Output":
                    roles = []
                    if _matches(self._residual, identity["full_name"]):
                        roles.append("residual_stream")
                    if _matches(self._heads, identity["full_name"]):
                        roles.append("output_head_logit")
                    if roles:
                        self.detail_buffer[-1]["role"] = ",".join(roles)

    @staticmethod
    def _extract_tensors_with_positions(data, prefix=""):
        if isinstance(data, torch.Tensor):
            return [(prefix, data)]
        tensors = []
        if isinstance(data, (tuple, list)):
            items = enumerate(data)
        elif isinstance(data, dict):
            items = data.items()
        else:
            return tensors
        for key, item in items:
            path = f"{prefix}_{key}" if prefix else str(key)
            tensors.extend(NumericalAnalyzer._extract_tensors_with_positions(item, path))
        return tensors

    @staticmethod
    def _compute_single_tensor_metrics(tensor):
        return tensor_metrics(tensor)

    @staticmethod
    def _aggregate_batch_metrics(per_batch_metrics):
        """Merge population moments over finite elements; preserve worst range."""
        if not per_batch_metrics:
            return {}
        result = {
            key: sum(m.get(key, 0) for m in per_batch_metrics)
            for key in (
                "numel",
                "finite_count",
                "nan_count",
                "inf_count",
                "posinf_count",
                "neginf_count",
            )
        }
        valid = [m for m in per_batch_metrics if m["finite_count"] > 0]
        n = result["finite_count"]
        if valid:
            mean = sum(m["mean"] * m["finite_count"] for m in valid) / n
            # Chan's merged population variance avoids cancellation from mean².
            var = (
                sum(m["finite_count"] * (m["std"] ** 2 + (m["mean"] - mean) ** 2) for m in valid)
                / n
            )
            result.update(
                mean=mean,
                std=math.sqrt(max(0.0, var)),
                max=max(m["max"] for m in valid),
                min=min(m["min"] for m in valid),
                max_diff=max(m["max_diff"] for m in valid),
                max_abs=max(m["max_abs"] for m in valid),
                rms=math.sqrt(sum(m["rms"] ** 2 * m["finite_count"] for m in valid) / n),
                l2_norm=math.sqrt(sum(m["l2_norm"] ** 2 for m in valid)),
                zero_ratio=sum(m["zero_ratio"] * m["finite_count"] for m in valid) / n,
            )
        else:
            result.update(
                {
                    key: float("nan")
                    for key in ("mean", "std", "max", "min", "max_diff", "rms", "zero_ratio")
                }
            )
            result.update(max_abs=0.0, l2_norm=0.0)
        result["batch_mode"] = True
        return result

    @torch.no_grad()
    def _compute_and_buffer(
        self, tensor, group, module_name, dtype, is_running_stat=False, source_loc="Unknown"
    ):
        if tensor.numel() == 0 or tensor.is_complex():
            return
        scope = "rank_local"
        if hasattr(tensor, "to_local") and hasattr(tensor, "placements"):
            scope = "local_shard" if any(p.is_shard() for p in tensor.placements) else "rank_local"
            tensor = tensor.to_local()
        # FSDP1 original parameters may be flattened partial views; do not label
        # pre-forward parameter measurements global, even if they look unsharded.
        if dtype == "Param" and scope == "rank_local":
            scope = "local_observed"
        use_batch = (
            dtype in ("Input", "Output", "Manual")
            and self.batch_size > 0
            and self.batch_dim is not None
            and tensor.ndim > 0
        )
        if use_batch:
            dim = self.batch_dim % tensor.ndim
            use_batch = tensor.shape[dim] == self.batch_size
        if use_batch:
            metrics = self._aggregate_batch_metrics(
                [tensor_metrics(part) for part in tensor.unbind(dim)]
            )
        else:
            metrics = tensor_metrics(tensor)
            metrics["batch_mode"] = False
        record = {
            "step": self.step_counter,
            "rank": self.rank,
            "scope": scope,
            "group": group,
            "module": module_name,
            "type": dtype,
            "source_loc": source_loc,
            "tensor_dtype": str(tensor.dtype).replace("torch.", ""),
        }
        record.update(metrics)
        if is_running_stat:
            record["running_mean" if "running_mean" in module_name else "running_var"] = metrics[
                "mean"
            ]
        self.detail_buffer.append(record)

    def step_forward(self, global_step):
        """Flush the current step; legacy callers may then omit begin_step()."""
        if not self.is_enable or not self.is_main_rank:
            return
        if self.detail_buffer and global_step != self.step_counter:
            raise ValueError(
                "step_forward must match the forward's step; call begin_step(step) before forward"
            )
        if self.detail_buffer:
            self._write_numerical_logs(global_step)
        if self.text_buffer and global_step % self.log_interval == 0:
            self._write_text_logs(global_step)
        self.detail_buffer.clear()
        self._reset_step_state()
        self.step_counter = global_step + 1

    step = step_forward

    def _write_text_logs(self, global_step):
        if self.active_save_dir:
            self._write_csv(f"text_logs_step_{global_step}.csv", self.text_buffer)
        self.text_buffer.clear()

    def _write_csv(self, name, records):
        directory = Path(self.active_save_dir)
        directory.mkdir(parents=True, exist_ok=True)
        columns = sorted({key for record in records for key in record})
        leading = [
            key
            for key in ("step", "group", "module", "type", "tensor_dtype", "source_loc")
            if key in columns
        ]
        columns = leading + [key for key in columns if key not in leading]
        with (directory / name).open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(records)

    def _write_numerical_logs(self, global_step):
        self.last_records = [dict(record) for record in self.detail_buffer]
        if self.active_save_dir:
            self._write_csv(f"layer_stats_step_{global_step}.csv", self.detail_buffer)
        if self.log_scalar_fn:
            output_ranges = {}
            for record in self.detail_buffer:
                path = f"{record['type']}/{record['group']}/{record['module']}"
                for metric in self.scalar_metrics:
                    if metric in record:
                        self.log_scalar_fn(f"{metric}/{path}", record[metric], global_step)
                if record["type"] == "Manual":
                    self.log_scalar_fn(
                        f"Max_Diff/Manual_{record['group']}/{record['module']}",
                        record["max_diff"],
                        global_step,
                    )
                if record["type"] == "Output" and math.isfinite(record["max_diff"]):
                    group = record["group"]
                    output_ranges[group] = max(output_ranges.get(group, 0.0), record["max_diff"])
            for group, value in output_ranges.items():
                self.log_scalar_fn(f"Max_Diff/Output_Group/{group}", value, global_step)
