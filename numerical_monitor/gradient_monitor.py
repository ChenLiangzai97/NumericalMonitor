"""Optimizer-boundary monitoring; no gradient hooks or training mutations."""

import csv
import math
from pathlib import Path
import re

import torch
import torch.distributed as dist

from .metrics import local_tensor, tensor_metrics


class GradientMonitor:
    """Monitor final accumulated gradients, parameters and optimizer updates.

    Call capture_gradients after backward/synchronization and AMP unscale, before
    clipping. Call before_optimizer_step immediately before the optimizer and
    after_optimizer_step immediately after, with updated=False on a skipped step.
    Every rank must call methods in the same order for sharded metrics. Explicit
    process_group is ONLY the shard group for ordinary FSDP1 original parameters;
    DDP replicas must not be summed. DTensor Shard placements are detected.
    """

    def __init__(
        self,
        model=None,
        optimizer=None,
        *,
        monitored_params=None,
        log_scalar_fn=None,
        save_dir=None,
        grad_interval=10,
        param_interval=100,
        optimizer_interval=100,
        update_interval=100,
        include=None,
        exclude=None,
        update_include=None,
        output_heads=(),
        shard_process_group=None,
        sharded=False,
        rank=None,
        eps=1e-12,
        adam_denominator=False,
    ):
        for interval in (grad_interval, param_interval, optimizer_interval, update_interval):
            if interval < 1:
                raise ValueError("Intervals must be positive")
        if eps <= 0:
            raise ValueError("eps must be positive")
        if monitored_params is None:
            if model is None:
                raise ValueError("model or monitored_params is required")
            monitored_params = dict(model.named_parameters())
        self.params = {}
        seen = set()
        for name, p in monitored_params.items():
            if id(p) in seen or (include and not re.search(include, name)):
                continue
            if exclude and re.search(exclude, name):
                continue
            if "_flat_param" in name:
                raise ValueError(
                    "FSDP1 layer metrics require use_orig_params=True or fsdp_grad adapters"
                )
            seen.add(id(p))
            self.params[name] = p
        self.optimizer = optimizer
        self.log_scalar_fn = log_scalar_fn
        self.save_dir = Path(save_dir) if save_dir else None
        self.grad_interval = grad_interval
        self.param_interval = param_interval
        self.optimizer_interval = optimizer_interval
        self.update_interval = update_interval
        self.update_include = update_include
        self.output_heads = tuple(output_heads)
        self.sharded = sharded
        self.shard_process_group = shard_process_group
        self.rank = rank if rank is not None else (dist.get_rank() if dist.is_initialized() else 0)
        self.is_main = self.rank in (-1, 0)
        self.eps = eps
        self.adam_denominator = adam_denominator
        self.enabled = True
        self.records = []
        self._snapshots = {}
        self._snapshot_step = None
        self._pending_optimizer_step = None
        self._pre_clip = None
        self._clip_events = self._clip_triggered = 0

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False
        self._snapshots.clear()
        self._snapshot_step = None
        self._pending_optimizer_step = None
        self._pre_clip = None

    def _emit(self, step, phase, layer, metric, value, scope="monitored_parameters"):
        if not self.is_main:
            return
        value = float(value)
        self.records.append(
            dict(step=step, phase=phase, layer=layer, metric=metric, value=value, scope=scope)
        )
        if self.log_scalar_fn:
            self.log_scalar_fn(f"{phase}/{layer}/{metric}", value, step)

    def _groups(self, parameter):
        if hasattr(parameter, "placements"):
            groups = []
            for axis, placement in enumerate(parameter.placements):
                if placement.is_partial():
                    raise ValueError("Partial DTensor must be redistributed before monitoring")
                if placement.is_shard():
                    groups.append(parameter.device_mesh.get_group(axis))
            return groups
        if self.sharded:
            if not dist.is_initialized():
                raise RuntimeError("Sharded monitoring requires an initialized process group")
            return [self.shard_process_group]
        return []

    @torch.no_grad()
    def _summaries(self, tensors):
        """Reduce sufficient statistics in batches by device and shard topology."""
        buckets = {}
        for name, (tensor, parameter) in tensors.items():
            groups = self._groups(parameter)
            base = local_tensor(parameter)
            if tensor is not None:
                value = local_tensor(tensor)
                x = value.to(
                    torch.float64 if value.dtype == torch.float64 else torch.float32
                ).reshape(-1)
            else:
                x = base.new_empty(0).float()
            if x.is_sparse:
                raise ValueError("Sparse tensors are not supported")
            finite = torch.isfinite(x)
            safe = torch.where(finite, x, 0.0)
            # Statistics buffer uses FP64 on CPU/CUDA to preserve large finite
            # gradient square sums. NPU uses FP32 for backend compatibility.
            acc = torch.float32 if x.device.type == "npu" else torch.float64
            sums = torch.stack(
                [
                    x.new_tensor(x.numel(), dtype=acc),
                    finite.sum().to(acc),
                    safe.to(acc).square().sum(),
                    torch.isnan(x).sum().to(acc),
                    torch.isinf(x).sum().to(acc),
                    x.new_tensor(tensor is not None, dtype=acc),
                ]
            )
            maximum = safe.abs().amax() if x.numel() else x.new_zeros(())
            key = (str(x.device), tuple(id(g) for g in groups))
            buckets.setdefault(key, (groups, []))[1].append((name, sums, maximum))
        result = {}
        for groups, rows in buckets.values():
            sums = torch.stack([r[1] for r in rows])
            maxima = torch.stack([r[2] for r in rows]).to(sums.dtype)
            for group in groups:
                dist.all_reduce(sums, op=dist.ReduceOp.SUM, group=group)
                dist.all_reduce(maxima, op=dist.ReduceOp.MAX, group=group)
            packed = torch.cat([sums, maxima[:, None].to(sums.dtype)], dim=1).cpu().tolist()
            for (name, _, _), values in zip(rows, packed):
                n, nf, sq, nan, inf, present, ma = values
                result[name] = dict(
                    numel=n,
                    finite_count=nf,
                    sum_sq=sq,
                    nan_count=nan,
                    inf_count=inf,
                    present=present > 0,
                    max_abs=ma,
                )
        return result

    @staticmethod
    def _merge(rows):
        return dict(
            numel=sum(r["numel"] for r in rows),
            finite_count=sum(r["finite_count"] for r in rows),
            sum_sq=sum(r["sum_sq"] for r in rows),
            nan_count=sum(r["nan_count"] for r in rows),
            inf_count=sum(r["inf_count"] for r in rows),
            max_abs=max((r["max_abs"] for r in rows), default=0),
        )

    @staticmethod
    def _norm(row):
        if row["nan_count"]:
            return math.nan
        if row["inf_count"]:
            return math.inf
        return math.sqrt(row["sum_sq"])

    def _layer_rows(self, rows):
        grouped = {}
        for name, row in rows.items():
            layer = name.rsplit(".", 1)[0] if "." in name else "root"
            grouped.setdefault(layer, []).append(row)
        return {layer: self._merge(parts) for layer, parts in grouped.items()}

    @torch.no_grad()
    def capture_gradients(self, step, *, force=False):
        """Read final, unscaled accumulated gradients; never unscale or clip."""
        if not self.enabled or (not force and step % self.grad_interval):
            return None
        rows = self._summaries({n: (p.grad, p) for n, p in self.params.items()})
        layers = self._layer_rows(rows)
        layers["global"] = self._merge(list(rows.values()))
        for layer, row in layers.items():
            self._emit(step, "Gradient", layer, "norm", self._norm(row))
            for metric in ("max_abs", "nan_count", "inf_count"):
                self._emit(step, "Gradient", layer, metric, row[metric])
        for name, row in rows.items():
            self._emit(step, "GradientParameter", name, "norm", self._norm(row))
            self._emit(step, "GradientParameter", name, "present", row["present"])
            for metric in ("max_abs", "nan_count", "inf_count"):
                self._emit(step, "GradientParameter", name, metric, row[metric])
        self._pre_clip = (step, layers)
        return layers

    @torch.no_grad()
    def record_clipping(self, step):
        """Observe after the caller clips; return post/pre norm (1 = unchanged).

        Frequency is over observed clipping events, not unsampled steps. Global
        norm clipping normally gives the same scale for every nonzero layer.
        """
        if not self.enabled:
            return None
        if self._pre_clip is None or self._pre_clip[0] != step:
            return None
        rows = self._summaries({n: (p.grad, p) for n, p in self.params.items()})
        post = self._layer_rows(rows)
        post["global"] = self._merge(list(rows.values()))
        ratios = {}
        for layer, before in self._pre_clip[1].items():
            old, new = self._norm(before), self._norm(post[layer])
            ratio = (
                new / old
                if old > 0 and math.isfinite(old)
                else (1.0 if old == new == 0 else math.nan)
            )
            ratios[layer] = ratio
            self._emit(step, "Clipping", layer, "scale", ratio)
            self._emit(step, "Clipping", layer, "post_norm", new)
        scale = ratios["global"]
        if math.isfinite(scale):
            self._clip_events += 1
            self._clip_triggered += scale < 1 - 1e-6
            self._emit(
                step,
                "Clipping",
                "global",
                "observed_frequency",
                self._clip_triggered / self._clip_events,
            )
        self._pre_clip = None
        return ratios

    def record_loss(self, loss, step, name="total", *, scope="rank_local"):
        """Observe supplied loss values; scalar input cannot reveal sample max."""
        if not self.enabled:
            return None
        stats = tensor_metrics(torch.as_tensor(loss))
        for key in ("mean", "max", "nan_count", "inf_count"):
            self._emit(step, "Loss", name, key, stats[key], scope)
        self._emit(step, "Loss", name, "numel", stats["numel"], scope)
        return stats

    def record_lr(self, step):
        if self.enabled and self.optimizer:
            for i, group in enumerate(self.optimizer.param_groups):
                self._emit(step, "Optimizer", f"group_{i}", "lr", group["lr"])

    @torch.no_grad()
    def before_optimizer_step(self, step):
        if not self.enabled:
            return
        if self._pending_optimizer_step is not None:
            raise RuntimeError("Finish the pending optimizer step before starting another")
        self._pending_optimizer_step = step
        self.record_lr(step)
        if step % self.param_interval == 0:
            rows = self._summaries({n: (p, p) for n, p in self.params.items()})
            for layer, row in self._layer_rows(rows).items():
                self._emit(step, "Parameter", layer, "norm", self._norm(row))
                self._emit(
                    step,
                    "Parameter",
                    layer,
                    "rms",
                    math.sqrt(row["sum_sq"] / max(row["finite_count"], 1)),
                )
                for key in ("nan_count", "inf_count"):
                    self._emit(step, "Parameter", layer, key, row[key])
            weight_rows = {n: row for n, row in rows.items() if n.rsplit(".", 1)[-1] == "weight"}
            for layer, row in self._layer_rows(weight_rows).items():
                if layer in self.output_heads:
                    self._emit(step, "OutputHead", layer, "weight_norm", self._norm(row))
        if step % self.update_interval == 0:
            self._snapshot_step = step
            self._snapshots = {
                n: local_tensor(p).clone()
                for n, p in self.params.items()
                if not self.update_include or re.search(self.update_include, n)
            }

    @torch.no_grad()
    def after_optimizer_step(self, step, *, updated=True):
        if not self.enabled:
            return
        if self._pending_optimizer_step != step:
            raise ValueError("Call before_optimizer_step with the same step first")
        self._emit(step, "Optimizer", "global", "updated", updated)
        if self._snapshot_step == step:
            before = self._summaries(
                {n: (old, self.params[n]) for n, old in self._snapshots.items()}
            )
            changes = self._summaries(
                {
                    n: (
                        (
                            local_tensor(self.params[n]).to(
                                torch.promote_types(old.dtype, torch.float32)
                            )
                            - old.to(torch.promote_types(old.dtype, torch.float32))
                        )
                        if updated
                        else torch.zeros_like(old),
                        self.params[n],
                    )
                    for n, old in self._snapshots.items()
                }
            )
            old_layers, delta_layers = self._layer_rows(before), self._layer_rows(changes)
            for layer in old_layers:
                numerator, denominator = (
                    self._norm(delta_layers[layer]),
                    self._norm(old_layers[layer]),
                )
                self._emit(
                    step, "Update", layer, "weight_ratio", numerator / (denominator + self.eps)
                )
                self._emit(step, "Update", layer, "norm", numerator)
            self._snapshots.clear()
            self._snapshot_step = None
        if step % self.optimizer_interval == 0 and self.optimizer:
            self.capture_optimizer_state(step)
        self._pending_optimizer_step = None

    @torch.no_grad()
    def capture_optimizer_state(self, step):
        """Read state after update; missing Adam state is explicitly unavailable."""
        if not self.enabled or self.optimizer is None:
            return
        for state_key in ("exp_avg", "exp_avg_sq"):
            tensors = {
                n: (self.optimizer.state.get(p, {}).get(state_key), p)
                for n, p in self.params.items()
            }
            rows = self._summaries(tensors)
            for name, row in rows.items():
                self._emit(step, "AdamParameter", name, f"{state_key}_available", row["present"])
            for layer, row in self._layer_rows(rows).items():
                rms = (
                    math.sqrt(row["sum_sq"] / row["finite_count"])
                    if row["finite_count"]
                    else math.nan
                )
                self._emit(step, "Adam", layer, f"{state_key}_rms", rms)
                self._emit(step, "Adam", layer, f"{state_key}_nan_count", row["nan_count"])
                self._emit(step, "Adam", layer, f"{state_key}_inf_count", row["inf_count"])
        if self.adam_denominator:
            self._capture_adam_denominator(step)

    def _capture_adam_denominator(self, step):
        tensors = {}
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                state = self.optimizer.state.get(p, {})
                key = "max_exp_avg_sq" if group.get("amsgrad") else "exp_avg_sq"
                if key not in state:
                    continue
                # This optional diagnostic may synchronize the optimizer step
                # scalar. It is sampled, never used to change the update.
                count = float(state.get("step", 0))
                correction = 1 - group["betas"][1] ** count
                if correction <= 0:
                    continue
                denominator = (local_tensor(state[key]) / correction).sqrt() + group["eps"]
                for name, candidate in self.params.items():
                    if candidate is p:
                        tensors[name] = (denominator, p)
        rows = self._summaries({n: tensors.get(n, (None, p)) for n, p in self.params.items()})
        for layer, row in self._layer_rows(rows).items():
            value = (
                math.sqrt(row["sum_sq"] / row["finite_count"]) if row["finite_count"] else math.nan
            )
            self._emit(step, "Adam", layer, "bias_corrected_denominator_rms", value)

    def flush(self):
        """Append CSV if configured, return consumed records and clear buffers."""
        if not self.is_main or not self.records:
            return []
        if not self.save_dir:
            records, self.records = self.records, []
            return records
        self.save_dir.mkdir(parents=True, exist_ok=True)
        path = self.save_dir / "training_metrics.csv"
        needs_header = not path.exists() or path.stat().st_size == 0
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["step", "phase", "layer", "metric", "value", "scope"]
            )
            if needs_header:
                writer.writeheader()
            writer.writerows(self.records)
        records, self.records = self.records, []
        return records

    close = flush
