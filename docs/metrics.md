# 监控指标与统计口径

`step` 统一指 optimizer step。梯度累积的多个 microbatch 共用同一 step；应在累积和梯度通信完成后采集最终梯度。

## 指标覆盖

| 优先级 | 指标 | 入口 / 字段 | 统计说明 |
|---|---|---|---|
| P0 | Loss Mean / Max | `record_loss()`，`Loss/.../mean,max` | 对传入的 loss 元素统计；标量输入的 mean 和 max 相同，不能还原样本级最大 loss |
| P0 | Global Grad Norm | `capture_gradients()`，`Gradient/global/norm` | 最终、反缩放后的全局 L2 梯度范数 |
| P0 | Per-Layer Grad Norm | `Gradient/{layer}/norm` | 按参数名父路径聚合；共享参数仅计算一次 |
| P0 | Parameter Norm | `before_optimizer_step()`，`Parameter/{layer}/norm` | 本次更新前参数 L2 范数 |
| P0 | Update / Weight Ratio | `after_optimizer_step()`，`Update/{layer}/weight_ratio` | 实际更新范数除以更新前参数范数加 eps；需要前后快照 |
| P0 | Activation RMS | `NumericalAnalyzer`，`rms` | 当前 rank 的有限激活元素 RMS |
| P0 | Activation Max Abs | `NumericalAnalyzer`，`max_abs` | 当前 rank 的有限激活元素最大绝对值 |
| P0 | NaN / Inf Count | `nan_count,inf_count,posinf_count,neginf_count` | 数值探针逐张量计数；梯度与优化器状态也有故障计数 |
| P0 | LR | `record_lr()` 或 `before_optimizer_step()` | 各 optimizer group 本次更新使用的 LR |
| P1 | Adam exp_avg RMS | `capture_optimizer_state()`，`exp_avg_rms` | 更新后的原始一阶矩状态，不做偏差修正 |
| P1 | Adam exp_avg_sq RMS | `exp_avg_sq_rms` | 更新后的原始二阶矩状态的 RMS；不是梯度 RMS |
| P1 | Gradient Max Abs | `Gradient/{layer}/max_abs` | 最终梯度最大绝对值；配合 NaN/Inf 计数判断 |
| P1 | Residual Stream RMS | `residual_modules` 或 `log_manual()` | 需明确选中残差相加后的输出；记录带 `residual_stream` role |
| P1 | Q/K Norm | `AttentionMonitor.observe()`，`q_norm,k_norm` | 每个 head 内各 token 的 L2 范数均值；另有 `q_rms,k_rms` |
| P1 | Attention Logit Max | `observe()` / `observe_logits()`，`logit_max` | 每个 head 的有限、未屏蔽 logit 最大值；采样时只代表采样行 |
| P1 | Attention Entropy | `entropy` | dropout 前概率的熵，单位 nats；对有效 query 行取均值 |
| P1 | LM Head Logit RMS | `output_head_modules`，`rms` | 输出记录带 `output_head_logit` role；E2E 可指定规划头或辅助头 |
| P1 | LM Head Weight Norm | `output_heads` 或数值探针输出头参数 | 梯度组件给层级范数；数值组件给逐参数 `l2_norm` 和 `output_head_weight` role |
| P1 | Clipping Ratio | `record_clipping()`，`scale,observed_frequency` | 记录裁剪后/前范数、采样事件中的触发频率，不是逐元素截断比例 |

`output_heads` 是 `GradientMonitor` 的精确层名列表；`output_head_modules` 是 `NumericalAnalyzer` 的正则列表，匹配 `group.module`。`Parameter/{layer}/norm` 包含该层 weight 与 bias；专用 `OutputHead/{layer}/weight_norm` 只统计名为 weight 的参数。所有 global 指标仅覆盖本监控器选中的参数集。

## 数值定义

- RMS：有限元素平方均值的平方根。
- L2 norm：有限元素平方和的平方根。`GradientMonitor` 的梯度和参数 norm 额外传播 NaN/Inf 故障状态。
- std：有限元素的总体标准差，分母是有限元素数，不采用样本修正。
- zero ratio：有限元素中零的占比。
- 全空或全非有限张量：分布统计无有效样本，使用 NaN，并保留元素数和故障计数。应结合 `finite_count` 解读。
- 启用 batch 拆分时，均值、RMS 和总体方差按有限元素数合并；`max_diff` 保留最坏单样本极差，因此可能不等于合并后 `max-min`。

实际更新比例为：

$$
r_t = \frac{\|\theta_{t+1}-\theta_t\|_2}{\|\theta_t\|_2+\varepsilon}
$$

其中，$r_t$ 是本次更新比例，$\theta_t$ 和 $\theta_{t+1}$ 分别是同一次 optimizer step 前后的参数，$\|\cdot\|_2$ 是 L2 范数，$\varepsilon$ 是防止除零的小常数。它包含优化器实际产生的更新，包括 weight decay；AMP 跳过更新时应传 `updated=False`。

Attention 熵为：

$$
H = -\sum_j p_j\log p_j
$$

其中，$H$ 是一个 query 行的注意力熵，$p_j$ 是该 query 对第 $j$ 个有效 key 的注意力概率，$\log$ 为自然对数。模块最终对每个 head 的有效 query 行取均值。全屏蔽行和包含数值故障的行不参与均值，但分别计数。

## 范围与采样

`NumericalAnalyzer` CSV 包含 `scope`、`rank`、`batch_mode`、`role`。`rank_local` 表示当前 rank 的激活；`local_shard` 表示本地 DTensor 分片；`local_observed` 参数表示 hook 当时观察到的局部值，不保证全局。

`GradientMonitor` 的参数与梯度统计使用正确分片组聚合；loss 默认仍为 `rank_local`，不会自动汇总不同 rank 的 loss。Attention 结果同样是 `rank_local`，并记录采样 query 的数量及是否覆盖全部 query。不要把采样最大值当成完整 Attention 的最大值。

建议的起始配置：梯度每 10 步，参数、Adam 状态和更新比例每 100 步，Attention 每 100 步且最多 32 个 query。频率是可调整起点，不能替代目标硬件上的吞吐和显存实测。
