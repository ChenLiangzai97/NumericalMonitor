# 验证记录

2026-11-09，在 Python 3.12、PyTorch 2.15.0+cpu 环境完成：

- `pytest -q`：43 项通过，1 项跳过。
- `ruff check .`：通过。
- `python examples/train_toy.py --steps 4`：真实 AdamW 训练、裁剪和 CSV 导出通过。
- `pip check`：无损坏依赖。

测试覆盖：有限值/非有限值、总体标准差、极大值和次正规数；梯度累积、共享参数、零与缺失梯度；实际 AdamW 更新（含 weight decay）、跳过更新、FP64 微小更新；Adam 状态与偏差修正分母；裁剪比例；步号与快照生命周期；checkpoint 重算；Attention 的 mask、causal、GQA、分块/采样及外层 autocast。

分片归约使用模拟通信验证了 SUM/MAX、正确通信组及缺失梯度时的数据类型一致性。真实双进程 Gloo 测试因执行环境不允许本地 TCP socket 而跳过，已保留在测试集供部署环境/CI 执行。

尚未验证 CUDA/NPU 吞吐与显存、真实 DTensor/FSDP 分布式训练、UTP 事件接入、各版本 FSDP1 私有回调。FSDP1 私有 Hook 为显式开启路径，不能把模拟 handle 测试等同于真实硬件适配。
