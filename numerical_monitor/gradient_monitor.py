"""在优化器更新边界采集梯度与状态，不修改训练张量或接管训练流程。"""

import csv
import math
from pathlib import Path
import re

import torch
import torch.distributed as dist

from .metrics import local_tensor, tensor_metrics


class GradientMonitor:
    """监控累积后的最终梯度、参数、Adam 状态和实际更新量。

    capture_gradients 必须在 backward、分布式同步和 AMP 反缩放完成后，
    梯度裁剪前调用；本组件不执行反缩放或裁剪。
    before_optimizer_step / after_optimizer_step 紧邻同一次更新，跳过更新时
    由训练框架传入 updated=False，避免把跳过的步骤误报成一次正常更新。

    分片统计要求各 rank 按一致顺序参与；仅日志写入限制在主 rank。
    普通 FSDP1 原始参数须显式指定 shard_process_group；DDP 副本不再累加。
    DTensor 根据 Shard 布局自动选择归约维度。
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
        # 按参数对象去重，避免共享权重经多个路径出现时放大全局范数。
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
        # 调用顺序校验独立于快照采样，非采样步也必须配对更新前后事件。
        self._pending_optimizer_step = None
        self._pre_clip = None
        self._clip_events = self._clip_triggered = 0

    def enable(self):
        """启用采集；分布式场景须在所有相关 rank 上同步切换。"""
        self.enabled = True

    def disable(self):
        """停止采集并释放参数快照；已产生的日志可另行 flush。"""
        self.enabled = False
        self._snapshots.clear()
        self._snapshot_step = None
        self._pending_optimizer_step = None
        self._pre_clip = None

    def _emit(self, step, phase, layer, metric, value, scope="monitored_parameters"):
        """只在主 rank 写入结果，不能用此处的限制包住上游分片归约。"""
        if not self.is_main:
            return
        value = float(value)
        self.records.append(
            dict(step=step, phase=phase, layer=layer, metric=metric, value=value, scope=scope)
        )
        if self.log_scalar_fn:
            self.log_scalar_fn(f"{phase}/{layer}/{metric}", value, step)

    def _groups(self, parameter):
        """只归约分片维度；复制维度上的同一份数据不能重复计入。"""
        if hasattr(parameter, "placements"):
            groups = []
            for axis, placement in enumerate(parameter.placements):
                if placement.is_partial():
                    # Partial 是待求和的中间表示，对它直接取平方和不等价于最终梯度范数。
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
        """按设备与分片通信组分桶，批量归约平方和、故障数和最大绝对值。

        不收集完整张量；缺失梯度仍占据固定统计位置，确保各 rank 的通信形状一致。
        平方和与计数用 SUM，最大绝对值用 MAX，不能混用同一种归约操作。
        """
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
            # CPU/CUDA 用 FP64 累加大梯度平方和；NPU 用 FP32 兼容其后端支持。
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
            # 即使某 rank 缺少 FP64 梯度，也要保持 MAX buffer 的跨 rank dtype 一致。
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
        """范数保留 NaN/Inf 故障信号，避免有限值子集掩盖训练异常。"""
        if row["nan_count"]:
            return math.nan
        if row["inf_count"]:
            return math.inf
        return math.sqrt(row["sum_sq"])

    def _layer_rows(self, rows):
        """按参数名父路径合并同层 weight、bias 等统计量。"""
        grouped = {}
        for name, row in rows.items():
            layer = name.rsplit(".", 1)[0] if "." in name else "root"
            grouped.setdefault(layer, []).append(row)
        return {layer: self._merge(parts) for layer, parts in grouped.items()}

    @torch.no_grad()
    def capture_gradients(self, step, *, force=False):
        """读取最终、已反缩放的累积梯度，记录全局、逐层和逐参数指标。"""
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
        """裁剪后观察梯度，记录裁剪后/前范数之比；1 表示尺度未变。

        触发频率的分母仅包含实际观测的采样事件，不包含未采样步骤。
        全局 norm clipping 下，各非零层通常具有相同缩放系数。
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
        """统计传入的 loss；已归约标量无法还原样本级最大 loss。"""
        if not self.enabled:
            return None
        stats = tensor_metrics(torch.as_tensor(loss))
        for key in ("mean", "max", "nan_count", "inf_count"):
            self._emit(step, "Loss", name, key, stats[key], scope)
        self._emit(step, "Loss", name, "numel", stats["numel"], scope)
        return stats

    def record_lr(self, step):
        """读取本次更新使用的各参数组 LR，须在 scheduler 改为下一步 LR 前调用。"""
        if self.enabled and self.optimizer:
            for i, group in enumerate(self.optimizer.param_groups):
                self._emit(step, "Optimizer", f"group_{i}", "lr", group["lr"])

    @torch.no_grad()
    def before_optimizer_step(self, step):
        """更新前采集参数，并在更新量采样步保存独立的本地参数快照。"""
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
            # 快照会占用额外内存；用 update_interval/update_include 限制频率与范围。
            self._snapshot_step = step
            self._snapshots = {
                n: local_tensor(p).clone()
                for n, p in self.params.items()
                if not self.update_include or re.search(self.update_include, n)
            }

    @torch.no_grad()
    def after_optimizer_step(self, step, *, updated=True):
        """更新后比较同一步的真实参数差值，包含优化器与 weight decay 的共同效果。"""
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
        """读取更新后的 Adam 原始矩状态；未初始化的状态标记为不可用而非零。"""
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
        """可选诊断：统计偏差修正、AMSGrad 与 eps 生效后的 Adam 分母尺度。"""
        tensors = {}
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                state = self.optimizer.state.get(p, {})
                key = "max_exp_avg_sq" if group.get("amsgrad") else "exp_avg_sq"
                if key not in state:
                    continue
                # 读取优化器步数可能引入设备同步，因此仅在低频诊断中启用。
                # 这里只解释更新尺度，不参与修改优化器的真实更新。
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
        """追加 CSV，返回并清空已消费记录；未配置文件输出时也释放日志缓存。"""
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
