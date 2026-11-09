# 训练接入与旧组件迁移

## 1. 原有数值监控接口

`NumericalAnalyzer` 继续接受原来的 `container={"net_parts": {...}}`、`groups`、`log_interval`、`save_dir`、`rank`、`batch_size`、`is_fsdp2`。原有 `enable/disable/is_log_step/log_text/log_manual/step_forward/step` 保留。

```python
from numerical_monitor import NumericalAnalyzer

analyzer = NumericalAnalyzer(
    trainer.container,
    log_scalar_fn=trainer.tb_logger.add_scalar,
    log_interval=100,
    rank=trainer.rank,               # 必须是 global rank
    batch_size=6,
    include_modules=[r"encoder\.blocks\.\d+$", r"planner\.head$"],
    residual_modules=[r"encoder\.blocks\.\d+$"],
    output_head_modules=[r"planner\.head$"],
    scan_parameters=False,          # 参数由 GradientMonitor 在更新前统计
)
analyzer.enable()
```

正则匹配 `group.module`。只有在模块输出确实是残差相加结果时，才应将其标为残差流；函数内部的残差可在模型中调用 `log_manual()`。输出头也需要明确指定，不会从名字猜测。

迁移时的行为变化：

- 实例彼此独立，不再使用进程内单例，避免训练和验证对象相互覆盖。
- 每次前向前调用 `begin_step(step)`，完成当前 step 后调用 `step_forward(step)`。这两处必须使用同一编号。
- hook 不再改写输出或安装 contiguous/reshard 防御。模型 storage 生命周期问题应独立定位，不与监控混合。
- 参数 hook 的结果明确标记为局部观察值；需要全局参数范数时使用 `GradientMonitor`。
- 默认保留同一模块的多次调用。`deduplicate_calls=True` 可以抑制 checkpoint 重算，也会丢弃合法的循环/共享调用，不能自动区分二者。
- UTP 的基类与事件参数没有包含在四份代码中，因此本包不伪造可直接替换的 `NumericalAnalyzerCallback`。保留项目原有 callback，将其中实例化导入改为本包，并按下面时机调用。

## 2. 梯度、AMP 与更新时机

以下是接入顺序，`scaler` 由训练框架负责创建和管理。此示例假定常规非 fused GradScaler 更新路径，优化器未在其他地方反缩放。

```python
analyzer.begin_step(step)           # 每个累积周期一次
optimizer.zero_grad(set_to_none=True)
for microbatch in accumulated_batches:
    with torch.autocast("cuda", dtype=torch.float16):
        loss = compute_loss(model, microbatch) / accumulation_steps
    scaler.scale(loss).backward()

# 这里须已经完成分布式梯度同步；不要在 DDP no_sync 区域内采集。
scaler.unscale_(optimizer)          # 每个 optimizer step 仅一次
monitor.capture_gradients(step)     # 原始、最终梯度，裁剪前
if use_clipping:
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    monitor.record_clipping(step)   # 裁剪后读数，不执行裁剪

monitor.before_optimizer_step(step)
# 框架应提供本次更新是否实际执行的结果；不要使用 optimizer.step() 返回值推断。
scaler.step(optimizer)
scaler.update()
monitor.after_optimizer_step(step, updated=optimizer_was_updated)
analyzer.step_forward(step)
monitor.flush()
```

`optimizer_was_updated` 由目标训练框架的 AMP/优化器适配提供。没有 GradScaler 的普通 BF16/FP32 训练，可直接 `optimizer.step()` 并使用默认 `updated=True`。不要为了监控重复执行 `unscale_`、裁剪或更新。

`before_optimizer_step()` 必须紧邻真实更新，避免 LR scheduler 把采集值改成下一步的 LR。若框架使用优化器 closure，在 closure 内重算前向/反向，以上前后快照方案不能直接套用；应由专门适配确保快照对应最终更新。多优化器应各自建立监控器，或明确分开监控的参数集。

裁剪频率只统计调用 `record_clipping()` 且有同 step 梯度快照的采样事件。`capture_gradients()` 未采样的 step 不会成为频率分母。全局 norm clipping 下各层缩放系数通常相同。

## 3. 分布式

| 模式 | 接入要求 |
|---|---|
| 单卡 / DDP | `GradientMonitor(model, optimizer)`；DDP 梯度同步后读取副本，不沿 world group 再求和 |
| FSDP2 / DTensor | 在分片状态、optimizer 边界采集；根据参数 DTensor 的 Shard placement 聚合，Replicate 维度不重复聚合；Partial placement 需要先由训练框架完成归约 |
| FSDP1 原始参数 | `use_orig_params=True`，`sharded=True`，传入 `shard_process_group`；所有 rank 使用一致的原始参数名和调用顺序 |
| FSDP1 FlatParameter | 使用下面独立的兼容采集器或额外层映射，不把 FlatParameter 直接称作某个原始层 |
| HSDP | 统计仅沿 shard group 求和，不能把已同步的 replica 再累加 |

FSDP 梯度、参数与 Adam 状态采集需要所有相关 rank 参与，只有写日志被限制在 rank 0。不要在外层 `if rank == 0:` 中包住这些方法。`NumericalAnalyzer` 和 Attention 的前向统计则可只在 rank 0 执行，因为它们不做分布式归约。

模型跨多个不相容的参数布局、非 DTensor 的张量并行、CPU offload、多种 shard group 混合时，需要先核验每个统计桶的布局；普通张量没有足够元数据自动推断这些配置。

## 4. 已有 FSDP 梯度组件

```python
from numerical_monitor.fsdp_grad import FSDPGradNormOptimizer

collector = FSDPGradNormOptimizer(
    monitored_params, param_to_groups, groups_info,
    top_k=10, process_group=shard_group, is_main_process=(rank == 0),
)
# 最终 backward / 同步 / AMP unscale 完成，且尚未裁剪：
work = collector.prepare()
# 可与不改变本次统计快照的工作重叠。
collector.finalize(work, step, writer.add_scalar, writer.add_text, logger)
```

保留 per-parameter、group、global norm 与 top-k，新增 max abs、NaN/Inf 计数，保留零梯度和缺失梯度。异步操作期间由 work 持有统计 buffer；不能重入覆盖。

`HSDPGradStatsHook` 使用 PyTorch FSDP1 私有回调，需要 `enable_private_hook=True` 显式开启；版本、handle 元数据和回调签名不兼容时会拒绝安装。它只适用于声明的单次 backward 合同，不应自动替代 AMP 反缩放后、梯度累积完成后的最终梯度统计。默认优先使用 `GradientMonitor` 或上面的 prepare/finalize 路径。生产环境必须验证卸载恢复和异常退出；没有在 GPU/NPU 的真实训练中验证前，不应视为已完成性能适配。

## 5. Attention 显式探针

```python
from numerical_monitor import AttentionMonitor

attention_monitor = AttentionMonitor(
    log_scalar_fn=writer.add_scalar,
    log_interval=100,
    query_chunk_size=32,
    max_query_samples=32,
)
# 放在模型内部，传入实际 RoPE / QK Norm 后的 Q/K：[..., H, L, D]
stats = attention_monitor.observe(
    "dit.block9.attention", q, k, step,
    attn_mask=mask, is_causal=False,
)
```

布尔 mask 中 `True` 表示可见；浮点 mask 为加性 bias，`-inf` 表示屏蔽。需传入真实 scale、bias 和 causal 配置，不能仅凭 Q/K 恢复所有模型特有逻辑。GQA 需显式 `enable_gqa=True`。若模型已暴露实际 logits，可调用 `observe_logits(name, logits, step, ...)`，避免重算 QK。

分块只限制临时 score 矩阵的 query 行数，未采样时算量仍随序列长度平方增长。采样结果是局部诊断，不等于完整 Attention 的全局最大值或完整熵均值。探针不禁用 FlashAttention，不要求模型返回完整权重。

## 6. 验证与关闭

验证前可关闭训练 analyzer，使用单独的验证 analyzer；每个验证 batch 设置明确的 step 编号，避免混用训练 optimizer step。验证无需调用梯度与优化器入口。

训练结束前先 `monitor.flush()`、`analyzer.step_forward(current_step)`，再 `analyzer.disable()`；`disable()` 会清除尚未写出的缓冲。私有 hook 使用后显式 `uninstall_patch()`，最好放在 `finally` 中。
