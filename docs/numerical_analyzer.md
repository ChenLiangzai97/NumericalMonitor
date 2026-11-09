# 数值监控接入

`NumericalAnalyzer` 沿用原组件的容器、启停、手动埋点和 CSV 接口。核心包不依赖 UTP、Transformers、TensorBoard 或 SwanLab；任意满足 `log_scalar_fn(tag, value, step)` 的日志函数都可以接入。

```python
from numerical_monitor import NumericalAnalyzer

monitor = NumericalAnalyzer(
    container={"net_parts": {"planner": model}},
    log_scalar_fn=writer.add_scalar,
    log_interval=10,
    save_dir="./numerical_logs",
    # 正则匹配 group.module，仅挂载需要的模块。
    include_modules=[r"planner\.blocks\.\d+$", r"planner\.head$"],
    # 只把明确位于残差相加后的模块输出标为残差流。
    residual_modules=[r"planner\.blocks\.\d+$"],
    output_head_modules=[r"planner\.head$"],
    scalar_metrics=["rms", "max_abs", "nan_count", "inf_count"],
    # 参数可交给 GradientMonitor 在优化器阶段统一采集。
    scan_parameters=False,
)
monitor.enable()
for optimizer_step in range(num_steps):
    monitor.begin_step(optimizer_step)
    # 同一个 optimizer step 的所有梯度累积 microbatch 共用一次 begin_step。
    predictions = model(inputs)
    loss = loss_fn(predictions, labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    monitor.step_forward(optimizer_step)
monitor.disable()
```

`begin_step()` 应在前向前调用，`step_forward()` 负责写出同一步记录；两者的 step 必须相同。旧的连续从 0 开始的调用仍可只在每步末尾调用 `step_forward()`。从 checkpoint 恢复时应显式使用 `begin_step()`。

## 统计含义

- `rms`、`max_abs`、`l2_norm`、`mean`、总体 `std`、`zero_ratio` 均只统计有限值；NaN 和正负 Inf 单独计数。全部为非有限值时，分布指标为 NaN。
- 激活记录的 `scope=rank_local`，只代表当前 rank 的数据。DTensor 分片记录为 `local_shard`；参数记录为 `local_observed` 时，也不保证已经收集全部 FSDP 参数。全局参数、梯度和优化器指标应使用 `GradientMonitor` 的分片聚合接口。
- 参数不按 batch 拆分。激活设置 `batch_size>0` 和明确的 `batch_dim` 后，匹配维度才做分样本统计。合并后的均值、总体方差、RMS 和 L2 norm 与整体有限值统计一致；`max_diff` 为最差样本内部极差，保留原组件口径。
- `role=residual_stream`、`output_head_logit` 和 `output_head_weight` 是显式配置的语义标记；不会根据模块名称猜测网络语义。
- 共享参数在每步只扫描一次。默认保留模块的所有调用，包括激活检查点重算；`deduplicate_calls=True` 每步只保留模块第一次输出，同时也会略过循环或共享模块的后续合法调用。需要区分这些情形时，应在训练适配层控制采集窗口。

## 业务手动埋点与过滤

纯函数和残差加法不会自动成为模块 Hook 事件，可以在已有张量处手动调用：

```python
if monitor.is_log_step():
    monitor.log_manual("planner", "block9.residual", residual)
    monitor.log_text("当前场景标识")
```

真值过滤不使用隐式猜测。调用者可以按输出路径注册过滤器：

```python
from numerical_monitor.tensor_filter import FilterRegistry

FilterRegistry.register(
    lambda position, tensor, module, identity: position.startswith("out_gt"),
    group="planner",
)
```

`save_dir=None` 只禁用 CSV，标量日志仍然工作，最近一步记录可通过 `last_records` 获取。`disable()` 会注销 Hook 并清空未写出的缓存；需要保留当前步时应先调用 `step_forward()`。

该实现将同一个张量的统计结果批量取回主机，但仍然存在每张量的设备同步。全模块、逐样本采集应低频开启；高频监控应限制模块和统计项，并在真实 GPU/NPU 训练环境测量开销。
