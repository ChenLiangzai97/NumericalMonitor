"""基于原 NumericalAnalyzer 扩展的运行时数值监控组件。

通过非侵入式的 PyTorch Forward Hook 机制，捕获并分析激活值、权重参数及
运行时 Buffer 的数值分布，保留动态 Hook 控制、递归张量提取、手动埋点和 CSV 接口。

前向记录只代表当前 rank 的 batch，不代表整个分布式训练批次。组件不发起集体通信、
不修改模型输出，也不保留计算图；仍处于 FSDP 分片状态的参数显式标记为 local_shard。
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
    """定义数值统计指标的标准键名集合（规避硬编码）。"""

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
    """将模块筛选规则统一编译为正则表达式列表。"""
    if isinstance(patterns, str):
        patterns = [patterns]
    return [re.compile(p) for p in (patterns or ())]


def _matches(patterns, name):
    """判断完整模块路径是否命中任一筛选规则。"""
    return any(p.search(name) for p in patterns)


class NumericalAnalyzer:
    """采样模块输入、输出和参数，保持模型执行结果不变。

    保留原构造参数以及 step()/step_forward() 接口。实例之间独立维护统计状态，
    允许训练、验证或多个模型分别配置监控，避免单例导致配置和缓存相互覆盖。

    Args:
        container: 模型实例容器字典，推荐 {'net_parts': {'encoder': model}}；
            也支持直接传入 nn.Module。
        log_scalar_fn: 标量日志写入回调函数，签名为 (tag, value, step)，
            如 TensorBoard 的 SummaryWriter.add_scalar。
        groups: 指定待监控的顶层大模块名称；为 None 时自动取 net_parts 全部 key。
        log_interval: 采样、统计计算与落盘的步长间隔，以 optimizer step 为单位。
        save_dir: 统计结果 CSV 文件的输出根路径；None 仅禁用写盘，不禁用标量日志。
        rank: 全局进程号；-1 时尝试从已初始化的分布式环境推断。
        batch_size: 大于 0 时允许对匹配 batch 维度的激活值分样本统计；不拆分参数。
        is_fsdp2: 保留的 FSDP2 兼容配置；本地分片统计不会被标为全局指标。
        include_modules: 匹配 group.module 的正则表达式；None 表示不限制模块。
        exclude_modules: 不注册 Hook 的模块正则表达式。
        residual_modules: 明确指定输出为残差流的模块，不根据名称猜测残差语义。
        output_head_modules: 明确指定输出为 head logits 的模块。
        scalar_metrics: 需要推送至标量日志的指标名称列表。
        batch_dim: 激活值的显式 batch 维度；None 表示禁用分样本统计。
        scan_parameters: 若参数已由 GradientMonitor 采集，可关闭前向参数扫描。
        deduplicate_calls: 每步只保留模块首次输出；可排除 checkpoint 重算，
            但也会跳过共享模块、循环模块在同一步内的后续合法调用。
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
        # 初始化数值明细与文本事件缓存；Hook 句柄分别管理，休眠时物理注销。
        self.hook_handles = []
        self.pre_hook_handles = []
        self.detail_buffer = []
        self.text_buffer = []
        self.last_records = []
        self.module_mapping = {}
        self.monitored_modules = {}
        # 同一模块和共享参数在一个 optimizer step 内只扫描一次；begin_step 时重置。
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
        # 默认全量监控策略：groups 未指定时，新加入 net_parts 的模块自动纳入监控。
        for name in groups if groups is not None else parts:
            module = parts.get(name)
            # 类型安全过滤：容器可能混入非 nn.Module 的采样器等业务对象。
            if isinstance(module, nn.Module):
                self.monitored_modules[name] = module
        self._build_mapping()

    def _build_mapping(self):
        """深度遍历网络拓扑，构建模块实例到业务组、模块名及代码位置的映射索引。"""
        for group_name, root in self.monitored_modules.items():
            for name, module in root.named_modules():
                layer = name or "root"
                full_name = f"{group_name}.{layer}"
                if self._include and not _matches(self._include, full_name):
                    continue
                if _matches(self._exclude, full_name):
                    continue
                # 共享模块只保留一个规范名称，避免同一实例被重复注册 Hook。
                self.module_mapping.setdefault(
                    module,
                    {
                        "group": group_name,
                        "module": layer,
                        "full_name": full_name,
                        "source_loc": None,
                    },
                )

    # =====================================================================
    # API 暴露层：启停控制与外部步进
    # =====================================================================
    def enable(self):
        """激活分析器，按 Case 隔离输出目录并重新注册前向监听 Hook。"""
        if not self.is_main_rank or self.is_enable:
            return
        self.is_enable = True
        # 实现批量多 Case 隔离路由：每个激活周期递增 Case 计数器。
        self.case_counter += 1
        self.active_save_dir = (
            str(Path(self.base_save_dir) / str(self.case_counter)) if self.base_save_dir else None
        )
        self._register_hooks()

    def disable(self):
        """休眠分析器：物理拔除 Hook 并清空未写出的缓存。

        如需保存当前步，调用方应先执行 step_forward()，再禁用监控。
        """
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
        """释放逐模块、逐参数去重状态，为下一 optimizer step 重置采集窗口。"""
        self._scanned_param_modules.clear()
        self._scanned_parameters.clear()
        self._seen_calls.clear()
        self._has_scanned_params_this_step = False

    def begin_step(self, step):
        """在前向前设置 optimizer step；同一步内多个梯度累积 microbatch 只调用一次。

        恢复训练时显式传入当前步号，避免前向采样和末尾写盘发生 Off-by-One 错位。
        """
        if step < 0:
            raise ValueError("step must be nonnegative")
        if self.detail_buffer:
            raise RuntimeError("flush the previous step with step_forward() first")
        self.step_counter = int(step)
        self._reset_step_state()

    def is_log_step(self, step=None):
        """暴露步长判定接口，供业务侧提前拦截非采样步的高开销旁路计算。"""
        current = self.step_counter if step is None else step
        return self.is_main_rank and self.is_enable and current % self.log_interval == 0

    @staticmethod
    def _source_location():
        """基于调用栈获取业务文件名与代码行号；模块 Hook 首次使用后缓存。"""
        for frame in inspect.stack()[2:]:
            if frame.filename != __file__ and "torch/nn/modules" not in frame.filename:
                return f"{Path(frame.filename).name}:{frame.lineno}"
        return "Unknown"

    def log_text(self, text, step=None):
        """高频文本事件接口：非记录步也可入队，在满足采样周期时统一写盘。"""
        if self.is_enable and self.is_main_rank:
            self.text_buffer.append(
                {
                    "step": self.step_counter if step is None else step,
                    "source_loc": self._source_location(),
                    "text": text,
                }
            )

    def log_manual(self, group_name, module_name, tensor):
        """主动式业务埋点，捕获 F.softmax、残差加法等无法由模块 Hook 感知的张量。"""
        if self.is_log_step() and isinstance(tensor, torch.Tensor):
            self._compute_and_buffer(
                tensor, group_name, module_name, "Manual", source_loc=self._source_location()
            )

    # =====================================================================
    # 核心引擎层：前向监听、参数扫描及张量提取
    # =====================================================================
    def _register_hooks(self):
        """挂载前向 Hook；pre-forward 逐模块扫描参数，post-forward 采集输入输出。"""
        if self.hook_handles:
            return
        for module in self.module_mapping:
            # Hook 返回 None，不替换模型输出，也不为监控发起 DTensor 全量收集。
            self.pre_hook_handles.append(module.register_forward_pre_hook(self._pre_forward_hook))
            self.hook_handles.append(module.register_forward_hook(self._forward_hook))

    def _pre_forward_hook(self, module, args):
        """当前模块执行前扫描其参数；不假设其他 FSDP 模块此时已完成 unshard。"""
        if not self.is_log_step() or not self.scan_parameters:
            return
        # 同一模块可能因共享调用、循环或 checkpoint 重算多次执行，每步只扫描一次。
        if id(module) not in self._scanned_param_modules:
            self._scanned_param_modules.add(id(module))
            self._scan_module_params_and_buffers(module)

    def _scan_module_params_and_buffers(self, module):
        """扫描单个模块自身的权重参数和运行时 Buffer，避免递归重复统计。"""
        identity = self.module_mapping.get(module)
        if identity is None:
            return
        # 按参数实例去重，兼容不同模块共享同一权重的情况。
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
        # 定向提取 Batch Normalization 等层维护的动态滑动均值与滑动方差。
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
        """保留全模块扫描兼容接口；只统计可见的本地值，不触发 FSDP 全量收集。"""
        for module in self.module_mapping:
            self._scan_module_params_and_buffers(module)

    def _forward_hook(self, module, inputs, output):
        """前向传播回调：执行采样拦截、调用位置缓存、张量过滤和多路激活提取。"""
        if not self.is_log_step():
            return
        if self.deduplicate_calls and id(module) in self._seen_calls:
            return
        self._seen_calls.add(id(module))
        identity = self.module_mapping[module]
        # 首次执行时回溯业务代码位置，后续直接复用缓存，减少 inspect.stack() 开销。
        if identity["source_loc"] is None:
            identity["source_loc"] = self._source_location()
        # 获取当前 group 的过滤链（全局默认 + group 专属）；真值等业务张量按规则排除。
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
                    # 残差流和输出 head 由配置显式标记，复用同一套 RMS 等张量统计。
                    roles = []
                    if _matches(self._residual, identity["full_name"]):
                        roles.append("residual_stream")
                    if _matches(self._heads, identity["full_name"]):
                        roles.append("output_head_logit")
                    if roles:
                        self.detail_buffer[-1]["role"] = ",".join(roles)

    @staticmethod
    def _extract_tensors_with_positions(data, prefix=""):
        """递归解析 Tuple/List/Dict，提取张量并以索引或键名保留其所在路径。

        跳过 None、字符串等非张量数据；单路输出保留模块原名，多路输出附加位置后缀。
        """
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
        """单张量统计兼容接口：复用有限值统计算子和独立 NaN/Inf 计数。"""
        return tensor_metrics(tensor)

    @staticmethod
    def _aggregate_batch_metrics(per_batch_metrics):
        """分 batch 统计结果聚合，输出与单张量统计同构的指标字典。

        max/min/max_diff 取各样本极值；均值、总体方差、RMS 和零值比例按有限元素量
        聚合，NaN/Inf 计数直接求和。max_diff 保留“最差样本内部极差”的原有口径，
        因此不一定等于所有样本合并后的 max-min。
        """
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
            # 使用 Chan 总体方差合并公式，避免直接计算 E[X²]-E[X]² 的大数相消。
            # 组间均值偏移也计入方差，不能对各样本标准差直接取平均。
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
        """对张量计算数值特征并存入明细缓存；不保留原张量或计算图。

        只允许激活张量按 batch 拆分，避免第一维恰等于 batch_size 的权重被误拆。
        分布式范围显式写入 scope，防止将当前 rank 的统计误认为全局统计。
        """
        if tensor.numel() == 0 or tensor.is_complex():
            return
        scope = "rank_local"
        if hasattr(tensor, "to_local") and hasattr(tensor, "placements"):
            scope = "local_shard" if any(p.is_shard() for p in tensor.placements) else "rank_local"
            tensor = tensor.to_local()
        # FSDP1 原始参数可能是展平后的局部视图；即使外观上未分片，前向采集值也不
        # 宣称全局完整。全局参数和梯度指标交给优化器阶段的分片聚合逻辑。
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
        """外部步进接口：写出当前步缓存，清理去重状态并推进下一步计数。

        兼容从 0 连续运行的旧调用方式；恢复训练等非连续步进应先调用 begin_step()。
        写盘步号必须对应本次前向，避免把一次采集结果误标到另一训练步。
        """
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

    # =====================================================================
    # 持久化落盘及标量日志
    # =====================================================================
    def _write_text_logs(self, global_step):
        """将缓存的文本事件统一写出；未配置 save_dir 时仅清理已消费事件。"""
        if self.active_save_dir:
            self._write_csv(f"text_logs_step_{global_step}.csv", self.text_buffer)
        self.text_buffer.clear()

    def _write_csv(self, name, records):
        """动态整理指标列并写入 CSV，使用 utf-8-sig 规避 Excel 中文解析乱码。"""
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
        """写出单层明细并按配置推送标量，保留原 Max_Diff 日志标签。

        最近一步记录独立保存在 last_records；关闭磁盘输出不影响标量日志。
        """
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
