# NumericalMonitor

基于现有 `NumericalAnalyzer` 和 FSDP 梯度监控组件扩展的 PyTorch 训练数值监控包。提供前向统计、梯度统计、优化器状态、实际参数更新和 Attention 探针。

## 安装与运行

```bash
python -m pip install -e .
python examples/train_toy.py --steps 8 --output numerical_logs/toy
```

基础依赖为 Python 3.10+、PyTorch 2.4+。FSDP1 私有接口随 PyTorch 版本变化，使用前必须在目标训练环境验证；FSDP2 需使用包含对应 DTensor/FSDP2 API 的 PyTorch 版本。

## 三个入口

| 组件 | 负责内容 | 接入位置 |
|---|---|---|
| `NumericalAnalyzer` | 激活 RMS/max abs、非有限值、局部参数统计、残差和输出头标记 | Module forward hook 或 `log_manual()` |
| `GradientMonitor` | 最终梯度、参数范数、Adam 状态、实际更新比例、loss、LR、裁剪统计 | backward 完成后与 optimizer 更新前后 |
| `AttentionMonitor` | 每个 head 的 Q/K 范数、logit 最大值、注意力熵 | 模型内部显式传入 Q/K 或已有 logits |

已有的 `FSDPGradNormOptimizer` 和 `HSDPGradStatsHook` 保留独立的兼容入口，见[训练接入](docs/integration.md)。

```python
from numerical_monitor import GradientMonitor, NumericalAnalyzer

analyzer = NumericalAnalyzer(model, log_interval=100, scan_parameters=False)
monitor = GradientMonitor(model, optimizer, grad_interval=10,
                          save_dir="numerical_logs/training")
analyzer.enable()

for step, batch in enumerate(loader):
    analyzer.begin_step(step)
    optimizer.zero_grad(set_to_none=True)
    loss = compute_loss(model, batch)
    monitor.record_loss(loss, step)
    loss.backward()
    # 使用梯度累积、分布式同步或 AMP 时，先完成它们，再采集最终梯度。
    monitor.capture_gradients(step)
    monitor.before_optimizer_step(step)
    optimizer.step()
    monitor.after_optimizer_step(step)
    analyzer.step_forward(step)
    monitor.flush()

analyzer.disable()
```

完整可运行示例：[examples/train_toy.py](examples/train_toy.py)。完整指标与口径：[docs/metrics.md](docs/metrics.md)。AMP、FSDP 与旧组件迁移：[docs/integration.md](docs/integration.md)。

## 统计范围与开销

- 前向探针只统计当前 rank 的输入输出，默认仅 rank 0 采集；不代表整个分布式 batch。
- `GradientMonitor` 在 DTensor 的 Shard 维度聚合统计；普通 DDP 参数保持副本语义，不重复求和。FSDP1 使用原始参数时，需显式提供 shard process group。
- 分布统计使用有限元素；NaN、正负 Inf 独立计数。梯度范数遇到 NaN/Inf 会保留故障信号，不能只看有限值统计。
- `GradientMonitor` 各类采样间隔独立。快照会增加内存占用；Attention 探针重算被采样的 query 行，不能视为零开销。
- 默认不修改模型输出、不裁剪梯度、不反缩放梯度、不执行优化器更新。`HSDPGradStatsHook` 的私有 patch 必须显式启用。

已在 CPU 验证统计结果、训练示例和分片归约路由。真实双进程 Gloo 测试因当前执行环境禁止本地 TCP socket 而跳过；GPU、NPU、实际 UTP 训练环境与各版本 FSDP1 私有 hook 仍需部署验证。
