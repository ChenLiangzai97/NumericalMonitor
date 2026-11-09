# NumericalMonitor 设计与集成接力文档

- 文档版本：v1.0.0；更新日期：2026-11-09。
- 对应实现：Python 包 v0.1.0，代码基线提交 `409193f85052aa6b64ae7103f298e871f3ad03a8`。
- 读者：接力集成到公司训练代码仓的本地 Agent 与维护者。
- 本文是集成总入口；后续修改指标定义、频率策略或训练事件映射时，应同步更新本文。

## 1. 当前状态与集成目标

目前已实现用户要求的 P0/P1 共 19 项指标的统计入口，以及原组件的基础统计和梯度 Top-K。已完成 CPU 正确性验证，尚未完成目标 GPU/NPU、真实 FSDP 与 UTP 的生产集成和性能验收。

获取仓库：`git clone https://github.com/ChenLiangzai97/NumericalMonitor.git`。开发环境可在仓库内执行 `python -m pip install -e '.[dev]'`；接入既有训练环境时，应遵循训练仓的依赖锁定，不因安装本包改变已经标定的 PyTorch 版本。若先以源码依赖集成，应记录使用的提交，便于复现。

本次集成建议采用：**每步轻量状态 + 每 1000 步完整诊断**。用户现网数值监控间隔为 1000 步，第一阶段保持这一频率，先验证统计口径和训练行为不变，再按实测结果逐项提高频率。

必须区分三件事：

| 状态 | 当前含义 |
|---|---|
| 已实现统计 | 对应指标有计算代码与接口，不代表训练仓已经调用 |
| 可作为实时候选 | 指标的数据量或复用条件适合每步观测，仍须验证目标硬件开销 |
| 已实现低开销实时通道 | **尚未实现**：按需计算、复用已有 norm、延迟批量回传和独立异步写盘仍是后续优化任务 |

本文中的“实时”指每个 optimizer step 采集；不承诺在同一步即时显示，也不表示所有指标全量开启。计算频率与日志发布频率应独立设计；当前接口尚未完整解耦二者。

## 2. 代码架构与职责

| 文件 / 对象 | 已实现职责 | 接力时的使用方式 |
|---|---|---|
| [metrics.py](../numerical_monitor/metrics.py) / `tensor_metrics()` | 有限值统计、总体方差、RMS、L2、NaN/Inf 计数；极大/极小值处理 | 共用统计算子，当前会计算整套指标 |
| [numerical_analyzer.py](../numerical_monitor/numerical_analyzer.py) / `NumericalAnalyzer` | 模块输入输出 Hook、局部参数/Buffer、嵌套张量提取、手动埋点、CSV | 前向和验证阶段；只在主 rank 采集，激活是 rank-local |
| [gradient_monitor.py](../numerical_monitor/gradient_monitor.py) / `GradientMonitor` | 最终梯度、参数、Adam 状态、实际更新量、loss、LR、裁剪统计 | 训练循环的 backward 后与 optimizer 更新前后 |
| [fsdp_grad.py](../numerical_monitor/fsdp_grad.py) / `FSDPGradNormOptimizer` | 保留 prepare/finalize 接口；逐参数/组/全局梯度、Top-K、max abs、NaN/Inf | FSDP1 梯度兼容路径，可独立接入，避免与其他采集器重复扫描 |
| 同文件 / `HSDPGradStatsHook` | FSDP1 私有后向回调采集与兼容性检查 | 显式开启；默认不作为集成首选 |
| [attention.py](../numerical_monitor/attention.py) / `AttentionMonitor` | 逐 head Q/K norm、RMS、logit max、entropy；分块与 query 抽样 | 模型内部显式传入 Q/K 或已有 logits |
| [tensor_filter.py](../numerical_monitor/tensor_filter.py) | 全局与业务组过滤链 | 显式排除 GT、mask 等不应混入激活统计的张量 |
| `examples/train_toy.py`、`tests/` | 可运行 CPU 示例和回归测试 | 本地先运行，再增加训练框架集成测试 |

默认不修改模型输出，不执行 backward、反缩放、裁剪或 optimizer.step。保留核心启停和步进 API，但不保证原 Callback 无修改替换，迁移要求见第 7 节。

## 3. 指标清单、定义、频率及实现入口

### 3.1 统一记号与统计范围

- `step`：一次 optimizer 更新机会的编号；AMP 跳过更新仍可记录该 step。多个梯度累积 microbatch 共用同一 step。
- `Layer`：`GradientMonitor` 按参数名的父路径聚合的层，通常是叶子模块；不自动等于整个 Transformer Block。
- `global`：本监控器选中的、按对象身份去重后的参数集合。启用 `include/exclude` 后，是选中集合的总量，不再代表全模型。
- 激活与 loss 默认只代表当前 rank；FSDP 参数/梯度统计需要按实际分片组汇总。
- 数值分布统计仅使用有限元素；NaN/Inf 单独计数。梯度和参数的专用 norm 则传播 NaN/Inf 故障状态，避免把异常掩盖成正常有限范数。

下表中的频率为**集成建议**，不是包默认值，也不是性能承诺。“关键位置每步”均指限制层数与张量范围后经过实测；当前代码没有专门的轻量算子。

### 3.2 P0 指标

| 指标 | 当前定义 / 粒度 | 时机与实现入口 | 频率建议及限制 |
|---|---|---|---|
| Loss Mean / Max | 传入 loss 张量有限元素的均值/最大值，另记故障数和元素数 | `record_loss(loss, step, name)`；`Loss/{name}/mean,max` | 每步候选，优先复用已有训练日志。标量 loss 的 mean=max，无法反推样本最大值；当前函数仍调用通用统计算子 |
| Global Grad Norm | 选中参数最终梯度拼接后的 L2 norm | 累积、通信、AMP 反缩放完成后，裁剪前 `capture_gradients(step)`；`Gradient/global/norm` | 训练已有此值时优先复用；当前接口会重新扫描，初始设 1000 步 |
| Per-Layer Grad Norm | 同层各参数梯度平方和合并后开方；共享权重只计一次 | 同上；`Gradient/{layer}/norm`，另有 `GradientParameter/{name}/norm` | 少数层每步/10 步候选；全层初始 1000 步 |
| Parameter Norm | 更新前同层参数拼接后的 L2 norm，包含该层 weight、bias | `before_optimizer_step(step)`；`Parameter/{layer}/norm` | 初始每 1000 步，通常不需要每步扫描 |
| Update / Weight Ratio | 同一次更新的真实参数差值 norm /（更新前参数 norm + eps），包含 weight decay | `before_optimizer_step` 保存快照；`after_optimizer_step` 比较；`Update/{layer}/weight_ratio` | 每 1000 步；有参数副本与差值开销，用 `update_include` 限制范围。不是 lr×grad 的近似值 |
| Activation RMS | 模块输入/输出有限激活元素的均方根 | 前向 Hook 或 `log_manual()`；记录字段 `rms` | 少数关键输出每步/10 步候选；全层输入输出保持 1000 步 |
| Activation Max Abs | 有限激活元素的最大绝对值 | 同上；字段 `max_abs` | 同 RMS；不能仅看它而忽略被分开计数的 Inf |
| NaN / Inf Count | 张量中 NaN、正负 Inf 的精确数量；同时保留有限元素数 | 数值记录 `nan_count/inf_count/posinf_count/neginf_count/finite_count`；梯度另有逐参数和逐层故障计数 | 关键张量每步候选；全网精确计数仍需完整扫描。每步仅判断是否存在异常是后续轻量通道设计，当前无独立只判断接口 |
| LR | 各 optimizer param group 本次更新使用的学习率 | `record_lr(step)`；更新前接口也会调用；`Optimizer/group_{i}/lr` | 可每步；在 scheduler 修改为下一步 LR 前读取，不要重复写入 |

### 3.3 P1 指标

| 指标 | 当前定义 / 粒度 | 时机与实现入口 | 频率建议及限制 |
|---|---|---|---|
| Adam exp_avg RMS | 更新后原始一阶矩状态的 RMS，按层合并，不作偏差修正 | `capture_optimizer_state(step)`；`Adam/{layer}/exp_avg_rms` | 每 1000 步；状态未初始化时标记 available=False，不能当作零 |
| Adam exp_avg_sq RMS | 更新后原始二阶矩张量本身的 RMS；不等于梯度 RMS，也不是简单对状态均值开方 | 同上；`exp_avg_sq_rms` | 每 1000 步；需额外读取一份与参数同规模的状态 |
| Gradient Max Abs | 最终梯度有限元素的最大绝对值，按参数/层/选中集合汇总 | `capture_gradients(step)`；`Gradient/{layer}/max_abs` | 少数层每步/10 步候选；全层初始 1000 步。兼容 FSDP 采集器的 max abs 会保留 NaN/Inf，需留意口径差异 |
| Residual Stream RMS | 明确的残差相加结果的 RMS，不是任意 FFN 输出的 RMS | `residual_modules` 匹配实际模块输出，或 `log_manual()` 传入残差张量 | 少数 Block 边界每步/10 步候选；必须确认位置，配置只负责语义标记 |
| Q/K Norm | 每个 head 内各 token 向量 L2 norm 的均值，跨本地 batch 合并；另有 Q/K 元素 RMS | `AttentionMonitor.observe(name, q, k, step, ...)`；`q_norm/k_norm/q_rms/k_rms` | 当前每 1000 步；此接口连同 logits/entropy 一起计算，没有 Q/K-only 快速通道 |
| Attention Logit Max | 实际 scale 和 mask 生效后，有限、未屏蔽 logits 的逐 head 最大值 | `observe()` 从 Q/K 重算，或 `observe_logits()` 读取已有 logits | 指定层，每 1000 步；可限制 query 抽样，不把采样最大值当作完整最大值 |
| Attention Entropy | softmax 后、dropout 前的注意力熵；逐 head 对有效 query 行取均值，单位 nats | 同上；`entropy` | 每 1000 步并抽样；全屏蔽行和含数值故障行排除，数量另记 |
| LM Head Logit RMS | 明确输出 head 的输出张量 RMS；E2E 可对应规划头或辅助头 | `output_head_modules`，记录带 `output_head_logit` role | 小输出每步/10 步候选；大词表或大轨迹候选输出须实测 |
| LM Head Weight Norm | 指定层名下名为 weight 的参数的 L2 norm，专用指标不包含 bias | `output_heads=(...)`；`OutputHead/{layer}/weight_norm`；数值 Hook 也可逐参数记录 | 每 1000 步；注意普通 `Parameter/{layer}/norm` 包含 bias |
| Clipping Ratio | 裁剪后/前梯度 norm；另记采样事件中的裁剪触发频率 | 裁剪前 `capture_gradients`，裁剪后 `record_clipping(step)`；`Clipping/{layer}/scale` | 复用现有裁剪系数适合每步；当前方法要再次扫描，初始随梯度采样。不是逐元素截断比例 |

### 3.4 保留或额外提供的统计

- 原数值组件：Max、Min、极差、Mean、总体 Std、零值比例、BN running mean/var；新增 `numel/finite_count` 等解释字段。
- FSDP 梯度兼容组件：逐参数/业务组/选中集合的 norm、平均参数 norm、Top-K 参数和故障计数。平均参数 norm 不等于全局 norm，也不等于元素 RMS。
- 通用梯度组件：缺失梯度的 `present` 标记；Adam 状态逐参数 `available` 标记；是否实际更新的 `Optimizer/global/updated`。
- 可选 `adam_denominator=True`：偏差修正、AMSGrad 和 eps 生效后的 Adam 分母 RMS，低频诊断使用；不是原始二阶矩 RMS。
- Attention：有效/全屏蔽/数值异常 query 数、NaN/Inf 数、实际 query 抽样数和统计来源。

### 3.5 关键数学定义与边界

对于待统计张量，令 F 为有限元素的索引集合，n 为其元素数，x_i 为第 i 个元素。有限值 RMS 定义为：

$$
\operatorname{RMS}(x)=\sqrt{\frac{1}{n}\sum_{i\in F}x_i^2}
$$

其中，$F$ 为有限元素索引集合，$n$ 为集合大小，$x_i$ 为第 $i$ 个元素。有限值 L2 norm 使用相同平方和但不除以 $n$；总体 Std 的平方和使用元素减去有限值均值后的偏差，并除以 $n$，不使用样本修正。

实际更新比例为：

$$
r_t=\frac{\|\theta_{t+1}-\theta_t\|_2}{\|\theta_t\|_2+\varepsilon}
$$

其中，$r_t$ 是第 $t$ 次更新的相对大小；$\theta_t$、$\theta_{t+1}$ 分别为同一次优化器更新前后的所选层参数；$\|\cdot\|_2$ 表示向量二范数；$\varepsilon$ 为防止除零的小常数。多参数层先合并平方和，不能对每个参数的比例取平均。AMP 跳过更新时传 `updated=False`，该步更新量记为零并单独标记跳过。

逐 query 的 Attention 熵为：

$$
H_i=-\sum_{j\in V_i}p_{ij}\log p_{ij}
$$

其中，$i$ 为 query 索引，$V_i$ 为该 query 允许访问的 key 集合，$j$ 为 key 索引，$p_{ij}$ 为 softmax 概率，$\log$ 为自然对数，$H_i$ 为该行的熵；逐 head 最终值是有效 query 的 $H_i$ 均值。

必须保留的边界口径：

1. `tensor_metrics()` 对空/全非有限输入的 mean/std/rms/min/max/zero_ratio 返回 NaN；有限 L2 与 max_abs 返回 0，并通过计数表示没有有限样本。`NumericalAnalyzer` 跳过空张量，不产生该条记录。
2. `GradientMonitor` 的梯度/参数 norm 有 NaN 则为 NaN，否则有 Inf 则为 Inf；其 max abs 与 RMS 类统计采用有限值子集。当前 `Parameter.rms` 在有限元素数为零时可能为 0，必须结合故障计数解读，不能把它当作健康信号；本地 Agent 可另行修正该边界并增加测试。
3. `FSDPGradNormOptimizer` 的兼容路径直接计算原梯度的平方和/极值，所以遇到非有限值时 norm/max abs 会保留故障值；不要与通用路径的有限值 max abs 混为同一口径。
4. 分 batch 时，均值、方差、RMS、L2 由样本统计正确合并；max_diff 保留最差样本内部极差，可能不等于全张量 max-min。仅激活按显式 batch 维度拆分，参数不拆。
5. `observe()` 将加性 mask 中的 -Inf 视为屏蔽；`observe_logits()` 约定输入已有的 -Inf 是结构性 mask。NaN、+Inf 等有效位置故障另记数，不参与正常行熵均值。
6. 缺失梯度与真实零梯度的 norm 都可能为零，依靠 `present` 区分；缺失的 Adam 状态不是已初始化的零状态。
7. 共享参数仅计一次；归属层使用其保留的规范名称，不给每个别名都重复贡献一份全局 norm。

## 4. 实时与间隔监控的设计

### 4.1 当前可用策略

| 层次 | 内容 | 集成起点 |
|---|---|---|
| 每步状态 | 训练已有 loss、LR、已有的梯度 norm/裁剪系数、AMP 更新是否跳过 | 尽量复用现有训练日志，不为这些标量再扫全网 |
| 关键位置趋势 | 少数残差/FFN/head 入口输出的 RMS、max abs、非有限值，以及关键层梯度 | 正确性与性能验证后从每 10 步起，逐项评估每步采集 |
| 周期诊断 | 全层输入输出、参数、Adam 状态、实际更新比例、采样 Attention | 起始每 1000 步，保留现有诊断节奏 |

若仍排查此前 L9 问题，关键位置可选 L9 FFN 输出、navi_token_mid、辅助 head 入口，以及 L9 FFN/head 的梯度；这些是建议位置，不是包内硬编码的模块名。

### 4.2 当前性能限制

- `tensor_metrics()` 每次计算整套统计，含多次张量扫描和临时张量。`scalar_metrics` **只控制推送字段，不控制计算字段**。
- 每个张量的浮点统计和整数计数分两次打包回传主机；batch 拆分会逐样本重复。尚无跨模块的延迟回传队列。
- Hook 可同时覆盖父子模块的输入输出，同一激活可能被多次统计；`include_modules/exclude_modules` 是当前有效的减负手段。
- 通用梯度路径每个布局桶同步执行 SUM/MAX，再回传 CPU，非写日志 rank 也执行统计和回传。
- 参数、梯度、裁剪后梯度、更新前快照、更新差值、Adam 两份矩状态全部采样时，通用统计入口一共执行 7 轮；可选 Adam 分母再增加一轮。
- CPU/CUDA 梯度路径使用 FP64 平方和中间量；NPU 使用 FP32。精确更新比例还持有参数副本及差值，均会增加峰值内存和带宽消耗。
- 新兼容路径没有完整保留原实现的预分配、双缓冲与设备侧 Top-K；`advance_buffer()` 为兼容空操作，不能把保留接口视为保留原实时性能。
- CSV 和标量回调在调用线程执行；`flush()` 批量写出不是后台异步写盘。Attention 分块只降低临时矩阵内存，不消除重算开销。

所以，**目前不应仅将间隔改为 1 就称为轻量实时监控**，也不能承诺固定的吞吐损失百分比。

### 4.3 后续实时优化任务，当前未实现

| 优先级 | 后续任务 | 验收方向 |
|---|---|---|
| 1 | 复用训练已有 loss/norm/clip 标量的入口 | 不额外扫描梯度或重复裁剪 |
| 2 | 独立计算字段配置；RMS/max abs/非有限标志的轻量路径 | 选择少量指标时不再计算 mean/std 等完整统计 |
| 3 | 设备侧跨模块累计，小型 buffer 批量回传 | 不在每个 forward Hook 立即等待 CPU 读数 |
| 4 | 预分配、异步分片归约、设备侧 Top-K 与流依赖 | 统计通信与训练重叠，buffer 不在通信完成前被覆盖 |
| 5 | 独立有界写盘队列、计算与发布频率分离 | 慢日志后端不阻塞训练，退出可排空，积压策略明确 |
| 6 | 异常触发后的短期详细采集窗口 | 各 rank 一致切换通信路径，不因 rank0 单独触发产生死锁 |

以上均为接力任务，不能调用尚不存在的“快速模式”参数或 API。Q/K-only、仅非有限标志、复用预计算梯度范数的专用入口当前也没有提供。

## 5. 集成事件与生命周期

### 5.1 训练步骤顺序

| 顺序 | 训练状态 | 监控动作 |
|---|---|---|
| 1 | optimizer step 开始，尚未前向 | `analyzer.begin_step(step)`，一个累积周期仅一次 |
| 2 | 各 microbatch 前向 | Hook 或显式 `log_manual()`；loss 保持原业务归约与缩放逻辑 |
| 3 | 所有 backward 与必要的梯度同步结束 | AMP 训练框架仅执行一次 unscale；监控器不代做 |
| 4 | 梯度已最终化，尚未裁剪 | `monitor.capture_gradients(step)` 或选定的兼容采集器 |
| 5 | 原训练框架执行裁剪后 | 需要时 `monitor.record_clipping(step)`；不重复调用裁剪 |
| 6 | optimizer 更新前，scheduler 尚未改为下一步 LR | `monitor.before_optimizer_step(step)` |
| 7 | 执行原 optimizer/scaler 更新 | 框架取得是否实际更新的结果 |
| 8 | 同次更新结束 | `monitor.after_optimizer_step(step, updated=...)` |
| 9 | 本步结束 | `analyzer.step_forward(step)`；按日志策略 `monitor.flush()` |

`before_optimizer_step` 与 `after_optimizer_step` 即使在非采样步也要同号配对。不要在 gradient accumulation 的中间 backward 统计“最终梯度”。不要用 optimizer.step 返回值是否为 None 判断 AMP 是否跳过更新。

### 5.2 当前 API 可用的低频初始化示例

以下使用现有 API；`model`、`optimizer`、`writer` 由目标训练框架提供。此配置用于接入起点，**不是实时优化版**。初始化时应已完成模型并行包装，尤其 FSDP2 需绑定最终 DTensor 参数及对应 optimizer。

```python
from numerical_monitor import AttentionMonitor, GradientMonitor, NumericalAnalyzer

analyzer = NumericalAnalyzer(
    model,
    log_scalar_fn=writer.add_scalar,
    log_interval=1000,
    save_dir="numerical_logs/forward",
    scan_parameters=False,  # 避免与优化器阶段重复扫描参数
    batch_size=0,           # 初始关闭逐样本重复计算，确有需要再开启
    # 直接传 nn.Module 时，Hook 名称前缀为 model；正则需要按实际模块名调整。
    include_modules=[r"model\.blocks\.\d+$", r"model\.head$"],
    residual_modules=[r"model\.blocks\.\d+$"],
    output_head_modules=[r"model\.head$"],
)
monitor = GradientMonitor(
    model,
    optimizer,
    log_scalar_fn=writer.add_scalar,
    save_dir="numerical_logs/training",
    grad_interval=1000,
    param_interval=1000,
    optimizer_interval=1000,
    update_interval=1000,
    output_heads=("head",),  # 精确层名，不是正则；须匹配实际参数名父路径
    adam_denominator=False,
    # FSDP1 原始分片参数还需 sharded=True、shard_process_group=实际分片组。
)
attention = AttentionMonitor(
    log_scalar_fn=writer.add_scalar,
    log_interval=1000,
    query_chunk_size=32,
    max_query_samples=32,
)
analyzer.enable()
```

这里的 residual 配置仅在 Block 输出确实为残差相加结果时成立。Attention 实例不会自动找到 Q/K，必须在模型内部显式调用；可先不接入，后续逐项开启。

包实际默认值是：NumericalAnalyzer 1000 步；GradientMonitor 梯度 10 步，参数/Adam/更新 100 步；AttentionMonitor 100 步、chunk=64、query 全量。**这些默认值未经目标训练环境性能标定，接力时应显式配置。**

### 5.3 普通 FP32/BF16 无 GradScaler 的接入骨架

下面是假定最终梯度已经同步、未使用 optimizer closure 的训练循环骨架；业务的取数、loss 和裁剪配置由训练仓提供。不要复制后新增一套本已存在的 backward/裁剪/更新流程，应将监控调用插入原流程。

```python
for step, batch in enumerate(loader, start=resume_step):
    analyzer.begin_step(step)
    optimizer.zero_grad(set_to_none=True)
    prediction = model(batch["input"])
    loss = loss_fn(prediction, batch["target"])
    monitor.record_loss(loss, step)  # 标量输入不包含逐样本最大值
    loss.backward()
    # 若有累积或异步通信，应在它们最终完成后执行以下采集。
    monitor.capture_gradients(step)
    if use_clipping:
        clip_gradients_in_training_framework()
        monitor.record_clipping(step)
    monitor.before_optimizer_step(step)
    optimizer.step()
    monitor.after_optimizer_step(step, updated=True)
    analyzer.step_forward(step)
    monitor.flush()
```

骨架中的 `clip_gradients_in_training_framework()` 代表现有框架的裁剪操作，不是本包 API。AMP/GradScaler 的正确顺序见 [integration.md](integration.md) 第 2 节；BF16 是否使用 scaler 以实际训练框架为准。

若希望准确监控每样本 loss max，须在业务 loss 归约前提供相应向量；多任务应分别记录业务 loss 与真正进入 backward 的总 loss，注明权重、mask、归约与累积缩放，不能将不同口径混成一条曲线。

## 6. 分布式、Attention 和日志范围

### 6.1 分布式契约

| 模式 | 当前要求 |
|---|---|
| 单卡/DDP | DDP 同步后的梯度是副本；不要再沿 world group 累加 |
| FSDP2/DTensor | optimizer 边界按 Shard placement 归约；Replicate 维不重复求和；Partial 必须先完成必要归约 |
| FSDP1 原始参数 | 使用 `use_orig_params=True`；显式传 `sharded=True` 与正确 shard process group；各 rank 参数名、集合和调用顺序一致 |
| FSDP1 FlatParameter | 通用入口拒绝 flat 参数名；应建立原参数映射/使用兼容路径。私有 Hook 还要求可用的原参数及 segment 元数据，不保证任意 FlatParameter 配置都能直接工作 |
| HSDP | 只在 shard group 内归约，不能把 replica group 上已同步的数据重复计入 |

所有参与分片统计的 rank 都必须执行同样的训练阶段入口与采样决策，不能把整个 `capture_gradients/before/after` 放进 `if rank == 0`。只有日志写入是 rank0-only。

NumericalAnalyzer 只在主 rank 采集激活，不进行 collective；AttentionMonitor 不内置 rank0 限制，集成方可仅在 rank0 调用，但记录仍必须标识 rank-local。rank0 激活正常不等于所有卡都没有异常。

FSDP 参数 Hook 看到的可能是局部分片、原始参数视图或当时可见值，不保证是全局完整参数；不为了监控强制 all-gather。全局参数/状态采用 optimizer 边界的分片统计。

多种不同 shard group、CPU offload、非 DTensor 张量并行或多设备模型应先核验布局与后端；不能靠某个布尔参数自动兼容所有混合并行组合。

### 6.2 Attention 接入要求

- Q/K 形状为 `[..., H, L, D]`：H 为 head 数、L 为 token 数、D 为每 head 向量维度；Q 与 K 的 token 数可以不同。
- 传入真实 RoPE、QK Norm 等变换后的 Q/K，以及实际 scale、mask、bias、causal 配置。布尔 mask 的 True 表示可访问，浮点 mask 是加性 bias。
- GQA 需显式 `enable_gqa=True`；head 数必须满足整除要求。
- 全量计算按 query 分块；确定性抽样保留原 query 索引，以正确处理 causal mask。只要抽样，logit max 和 entropy 都不代表完整序列精确统计。
- 当前 `observe()` 诊断重算关闭外层 autocast，使用 FP32；`observe_logits()` 可复用已暴露的真实 logits。两者都不能宣称融合 Attention 内部结果可无成本直接获取。

### 6.3 输出与可解释性字段

| 输出 | 位置 / 字段 | 使用要求 |
|---|---|---|
| 前向明细 CSV | `save_dir/{case}/layer_stats_step_{step}.csv` | case 随 enable 激活周期递增；scope/rank/role/batch_mode 表达范围 |
| 文本事件 CSV | 同 case 下 `text_logs_step_{step}.csv` | 文本可跨非采样步缓存，于落盘采样步统一消费 |
| 梯度/优化器 CSV | `save_dir/training_metrics.csv` | 长表列：step、phase、layer、metric、value、scope |
| 标量回调 | `log_scalar_fn(tag, value, step)` | 可接 TensorBoard/SwanLab 等适配器；包不自带对应服务 |
| Attention 返回值 | `scope=rank_local`、`source`、`sampled`、query 数、`per_head` | 返回值按需求另行持久化；配置 callback 后可输出标量，本身不写独立 CSV |

`GradientMonitor.flush()` 返回并清空记录；无 save_dir 时也消费缓存。`NumericalAnalyzer.save_dir=None` 只关闭 CSV，标量仍会输出，最近一次结果在 last_records。数值组件 disable 会清除未消费的缓存，应先完成当前 step 的输出。长期不 flush 会累积训练记录；文件写出目前是同步的。

## 7. 从原组件迁移的要点

1. 尽量保留原代码注释，尤其业务背景和同步语义；失效注释必须随行为修正，新增逻辑补充必要中文说明。仓库已有 [AGENTS.md](../AGENTS.md) 约定。
2. NumericalAnalyzer 的容器、启停、manual、step 等 API 保留，但移除了进程内单例。原业务若通过“再次无参实例化”查找全局对象，需要改为显式传递同一个实例。
3. 原 NumericalAnalyzerCallback 未作为可直接替换对象打包。保留训练仓 callback 外壳，核对其事件真正发生位置后再调用本包，不凭事件名称推断它在裁剪前还是后。
4. 新版不再使用会改写输出的 contiguous/reshard guard；原训练仓若依赖该行为，应独立分析 storage 生命周期与修复归属，不因替换监控器而静默丢弃仍需的模型修复。
5. checkpoint 重算默认会重复记录激活。`deduplicate_calls=True` 只保留每模块每步第一次调用，同时会抑制合法共享/循环调用；不是准确识别重算的自动方案。
6. FSDPGradNormOptimizer 保留 prepare/finalize 与原标签入口，但新增指标、共享参数和零梯度处理后，平均值/性能特征可能与旧版不同。先核对既有看板口径，再切换。
7. HSDPGradStatsHook 需显式 `enable_private_hook=True`；仅声明支持每步一次同步 backward，不能直接套入梯度累积。安装前有签名/元数据检查，退出要在 finally 中卸载。模拟 handle 测试不等价于生产 FSDP1 适配。
8. 多优化器为各自参数集合创建独立监控器；optimizer closure、FSDP 延迟归约、梯度压缩及特殊优化器的状态布局需要单独适配。
9. 当前 forward Hook 默认只采集位置参数中的输入张量；通过 kwargs 传入的张量需 `log_manual()` 或由本地 Agent 扩展 Hook 适配，不能宣称已覆盖全部输入。

## 8. 本地 Agent 的接力任务与完成标准

### 阶段 A：确认事件并保持 1k 诊断

- 列出训练仓 PyTorch、设备、并行方式、mesh/process group、混合精度、GradScaler、累积步数、checkpoint 和 optimizer 配置。
- 对照真实调用链确认 forward、最终梯度同步、unscale、clip、optimizer、scheduler 各阶段位置，形成事件映射。
- 先接入每 1000 步的数值、梯度、参数与 Adam 统计，按需开启实际更新快照；避免旧监控与新版重复扫描。
- 校验参数规范名与 Hook 的 group.module 名称；GT 与合法 mask 从业务激活统计中显式排除。
- 用固定随机种子/相同输入验证开启监控前后输出、loss、梯度和更新一致，监控不应改变训练行为。

### 阶段 B：正确性与分布式测试

- 保留并运行现有 CPU 测试；新增目标训练 callback 的接入测试。
- 真正至少两 rank 验证分片 norm 对齐集中式参考，HSDP 不随 replica 数重复放大；含空分片、缺失梯度、共享权重和非有限值。
- 验证梯度累积/AMP/裁剪的读取时机，AMP 跳过更新、训练恢复的 step 对齐、评估切换和最后一步日志排空。
- 实际更新比例用独立更新前后参数差值比对；Adam 状态数值与 optimizer.state 对齐。
- Attention 用真实 mask/scale/位置变换核对小规模显式实现，确认抽样与全量结果的差异被标记。

### 阶段 C：性能验收与轻量通道

对同一训练配置与输入分别测：关闭监控、现有每 1k 步、候选每步轻量通道。充分预热后覆盖多个采样周期，统计平均/P50/P95 step 耗时、吞吐、各 rank 耗时、设备峰值显存和日志积压；报告采样步和普通步，避免平均值掩盖卡顿。不要在每层加入全设备 synchronize 来测时；用 profiler/events 归因，在必要测量边界同步。

额外耗时比例定义为：

$$
\eta=\frac{T_{\mathrm{on}}-T_{\mathrm{off}}}{T_{\mathrm{off}}}
$$

其中，$\eta$ 为额外耗时比例，$T_{\mathrm{on}}$ 为开启指定监控方案后的平均 step 时间，$T_{\mathrm{off}}$ 为同条件关闭监控时的平均 step 时间。训练吞吐、P95 和显存也必须同时报告。可以先讨论采用不高于 3% 的额外耗时预算，但这是待确认的验收目标，**不是当前实现达成的结果**。

按第 4.3 节逐项优化后，再决定哪些关键张量每步采集。优化需保留指标口径和 fault 信号，不能用抽样近似悄悄替代声称精确的全局统计。

### 阶段 D：交付

交付训练仓接入提交、实际调用位置、明确的指标开关/频率/层范围配置、目标硬件正确性与性能报告、已验证并行组合及剩余限制。更新本文、指标文档和版本变更记录；不要将“代码有接口”当作“生产实时监控已验收”。

## 9. 已有验证证据与相关文档

- 当前 CPU 结果：43 项测试通过，1 项真实 Gloo 测试因执行环境禁止本地 TCP socket 跳过；训练示例与静态检查通过。
- 未验证：CUDA/NPU 吞吐与峰值显存、真实 DTensor/FSDP/HSDP 训练、UTP callback 事件契约及所有版本的 FSDP1 私有回调。
- [验证记录](validation.md)、[指标字段参考](metrics.md)、[训练接入补充](integration.md)、[数值组件细节](numerical_analyzer.md)。
- 可运行示例与基线命令：`python examples/train_toy.py --steps 4`、`pytest -q`、`ruff check .`。示例只验证 API 使用和 CPU 数值结果，不是目标硬件性能基准。
