"""通过显式埋点采集 Attention 指标，不修改模型的前向路径。

Q/K 形状为 ``[..., heads, tokens, head_dim]``，依次表示批次维、注意力头数、
序列长度和每个头的特征维度。重算分数有额外开销，应仅在指定层和采样步调用；
本组件不会关闭 FlashAttention，也不会强制模型返回注意力权重。
"""

import math
from typing import Callable, Optional

import torch


class AttentionMonitor:
    """按注意力头统计 Q/K 尺度、有限 logit 最大值及注意力熵。

    熵使用自然对数和 dropout 前的概率。全遮蔽行及含数值异常的 query 行
    不参与平均熵；异常计数仅覆盖未遮蔽位置。所有结果均为当前 rank 的激活统计，
    不代表跨卡全局统计。

    Args:
        log_scalar_fn: 可选标量回调，签名为 ``callback(tag, value, step)``。
        log_interval: 采集间隔，单位为 optimizer step。
        query_chunk_size: 每个临时分数矩阵最多包含的 query 行数。
        max_query_samples: 等距、确定性抽样的 query 数量上限。为 None 时分块计算
            所有 query；分块限制临时内存，但全量自注意力的计算量仍随序列长度平方增长。
    """

    def __init__(
        self,
        log_scalar_fn: Optional[Callable] = None,
        log_interval: int = 100,
        query_chunk_size: int = 64,
        max_query_samples: Optional[int] = None,
    ):
        if log_interval < 1 or query_chunk_size < 1:
            raise ValueError("log_interval and query_chunk_size must be positive")
        if max_query_samples is not None and max_query_samples < 1:
            raise ValueError("max_query_samples must be positive or None")
        self.log_scalar_fn = log_scalar_fn
        self.log_interval = log_interval
        self.query_chunk_size = query_chunk_size
        self.max_query_samples = max_query_samples

    def is_log_step(self, step):
        """判断当前 optimizer step 是否需要采集。"""
        return step % self.log_interval == 0

    def _query_indices(self, length, device):
        """返回原序列中的 query 索引；抽样后仍保留原位置以正确应用 causal mask。"""
        count = min(length, self.max_query_samples or length)
        if count == length:
            return torch.arange(length, device=device)
        return torch.linspace(0, length - 1, count, device=device).long()

    @staticmethod
    def _validate_mask(attn_mask, shape, device):
        """校验 mask 并广播为分数矩阵形状，广播视图不复制完整矩阵。"""
        if attn_mask is None:
            return None
        if attn_mask.device != device:
            raise ValueError("attn_mask must be on the same device as Q/K or logits")
        if attn_mask.dtype != torch.bool and not attn_mask.is_floating_point():
            raise TypeError("attn_mask must be boolean or floating point")
        try:
            return torch.broadcast_to(attn_mask.detach(), shape)
        except RuntimeError as exc:
            raise ValueError("attn_mask is not broadcastable to attention scores") from exc

    @staticmethod
    def _apply_mask(logits, mask, indices, is_causal):
        """对当前 query 分块应用布尔或加性 mask，返回分数及允许参与统计的位置。"""
        allowed = torch.ones_like(logits, dtype=torch.bool)
        if mask is not None:
            chunk_mask = mask.index_select(-2, indices)
            if chunk_mask.dtype == torch.bool:
                allowed &= chunk_mask
            else:
                # -inf 表示结构性遮蔽；NaN 和 +inf 仍视为数值异常。
                allowed &= ~torch.isneginf(chunk_mask)
                logits = logits + chunk_mask.to(dtype=logits.dtype)
        if is_causal:
            # 非方形分数矩阵沿左上角对齐，与 SDPA 的 causal mask 约定一致。
            keys = torch.arange(logits.shape[-1], device=logits.device)
            allowed &= keys[None, :] <= indices[:, None]
        return logits, allowed

    @staticmethod
    def _new_accumulator(heads, device):
        """创建逐 head 累计器，用于合并各个 query 分块的统计结果。"""
        return {
            "logit_max": torch.full((heads,), -torch.inf, device=device),
            "entropy_sum": torch.zeros(heads, device=device),
            "valid_query_count": torch.zeros(heads, dtype=torch.int64, device=device),
            "fully_masked_query_count": torch.zeros(heads, dtype=torch.int64, device=device),
            "invalid_query_count": torch.zeros(heads, dtype=torch.int64, device=device),
            "nan_count": torch.zeros(heads, dtype=torch.int64, device=device),
            "inf_count": torch.zeros(heads, dtype=torch.int64, device=device),
        }

    @staticmethod
    def _accumulate(acc, logits, allowed):
        """累计有限分数极值、有效行熵及异常计数；跨批次维聚合但保留 head 维。"""
        heads, queries, keys = logits.shape[-3:]
        logits = logits.reshape(-1, heads, queries, keys)
        allowed = allowed.reshape(-1, heads, queries, keys)
        finite = torch.isfinite(logits)
        faults = allowed & ~finite
        eligible = allowed & finite
        finite_logits = torch.where(eligible, logits, -torch.inf)
        acc["logit_max"] = torch.maximum(acc["logit_max"], finite_logits.amax(dim=(0, 2, 3)))
        has_key = allowed.any(-1)
        fault_row = faults.any(-1)
        valid_row = has_key & ~fault_row
        # 避免对全 -inf 行执行 softmax；全遮蔽行和异常行不参与平均熵。
        safe = torch.where(valid_row[..., None], finite_logits, 0.0)
        log_probs = torch.log_softmax(safe, dim=-1)
        probs = log_probs.exp()
        finite_log_probs = torch.where(torch.isfinite(log_probs), log_probs, 0.0)
        entropy = -(probs * finite_log_probs).sum(-1)
        acc["entropy_sum"] += torch.where(valid_row, entropy, 0.0).sum(dim=(0, 2))
        acc["valid_query_count"] += valid_row.sum(dim=(0, 2))
        acc["fully_masked_query_count"] += (~has_key).sum(dim=(0, 2))
        acc["invalid_query_count"] += (has_key & fault_row).sum(dim=(0, 2))
        acc["nan_count"] += (allowed & torch.isnan(logits)).sum(dim=(0, 2, 3))
        acc["inf_count"] += (allowed & torch.isinf(logits)).sum(dim=(0, 2, 3))

    def _finish(self, name, acc, step, metadata, extra=None):
        """合并采样元数据并输出；没有有效行时以 NaN 表示熵不可用。"""
        count = acc.pop("valid_query_count")
        entropy_sum = acc.pop("entropy_sum")
        acc["entropy"] = torch.where(count > 0, entropy_sum / count.clamp_min(1), torch.nan)
        acc["valid_query_count"] = count
        acc["logit_max"] = torch.where(acc["logit_max"] == -torch.inf, torch.nan, acc["logit_max"])
        if extra:
            acc.update(extra)
        # 按浮点数和整数分别批量传回 CPU，保留计数精度，
        # 同时避免要求设备支持 FP64（部分加速器不支持）。
        count_keys = {
            "valid_query_count",
            "fully_masked_query_count",
            "invalid_query_count",
            "nan_count",
            "inf_count",
        }
        metrics = {}
        for is_count in (False, True):
            names = [key for key in acc if (key in count_keys) == is_count]
            if names:
                matrix = torch.stack([acc[key] for key in names]).cpu().tolist()
                metrics.update(dict(zip(names, matrix)))
        result = {
            "name": name,
            "step": step,
            "scope": "rank_local",
            **metadata,
            "per_head": metrics,
        }
        if self.log_scalar_fn is not None:
            for metric, values in metrics.items():
                for head, value in enumerate(values):
                    self.log_scalar_fn(f"Attention/{name}/head_{head}/{metric}", value, step)
        return result

    @torch.no_grad()
    def observe(
        self,
        name,
        query,
        key,
        step,
        attn_mask=None,
        is_causal=False,
        scale=None,
        enable_gqa=False,
    ):
        """从显式传入的 Q/K 中，以 FP32 重算选定 query 行的注意力分数。

        模型适配代码应传入实际参与 Attention 的 Q/K，即已完成 RoPE、归一化等
        模型特定变换的张量。布尔 mask 的 True 表示允许关注，与 SDPA 一致；
        浮点 mask 直接加到分数上，-inf 表示遮蔽。GQA 需显式开启
        ``enable_gqa=True``，且 Q 头数必须是 K 头数的整数倍。
        """
        if not self.is_log_step(step):
            return None
        if query.ndim < 3 or key.ndim != query.ndim:
            raise ValueError("Q/K must have shape [..., heads, tokens, head_dim]")
        if query.shape[:-3] != key.shape[:-3]:
            raise ValueError("Q/K batch dimensions must match")
        if query.shape[-1] != key.shape[-1] or query.shape[-1] == 0:
            raise ValueError("Q/K must have the same positive head dimension")
        if query.device != key.device:
            raise ValueError("Q/K must share a device")
        if not query.is_floating_point() or not key.is_floating_point():
            raise TypeError("Q/K must be floating point")
        if any(size == 0 for size in (*query.shape, *key.shape)):
            raise ValueError("Q/K must be nonempty")
        q = query.detach().float()
        k = key.detach().float()
        original_key_heads = k.shape[-3]
        if q.shape[-3] != k.shape[-3]:
            if not enable_gqa or q.shape[-3] % k.shape[-3]:
                raise ValueError("Unequal head counts require enable_gqa and divisible Q/K heads")
            k = k.repeat_interleave(q.shape[-3] // k.shape[-3], dim=-3)
        heads, length, dim = q.shape[-3:]
        key_length = k.shape[-2]
        score_shape = q.shape[:-3] + (heads, length, key_length)
        mask = self._validate_mask(attn_mask, score_shape, q.device)
        factor = float(scale) if scale is not None else 1 / math.sqrt(dim)
        if not math.isfinite(factor):
            raise ValueError("scale must be finite")
        indices = self._query_indices(length, q.device)
        acc = self._new_accumulator(heads, q.device)
        kt = k.transpose(-1, -2)
        # 仅调用 float() 无法阻止外层 autocast 再次降低 matmul 的精度；
        # 显式关闭 autocast，确保诊断重算始终使用 FP32。
        with torch.autocast(device_type=q.device.type, enabled=False):
            for chunk in indices.split(self.query_chunk_size):
                logits = torch.matmul(q.index_select(-2, chunk), kt) * factor
                logits, allowed = self._apply_mask(logits, mask, chunk, is_causal)
                self._accumulate(acc, logits, allowed)
        flat_q = q.reshape(-1, heads, length, dim)
        flat_k = k.reshape(-1, heads, key_length, dim)
        extra = {
            "q_norm": torch.linalg.vector_norm(flat_q, dim=-1).mean(dim=(0, 2)),
            "k_norm": torch.linalg.vector_norm(flat_k, dim=-1).mean(dim=(0, 2)),
            "q_rms": flat_q.square().mean(dim=(0, 2, 3)).sqrt(),
            "k_rms": flat_k.square().mean(dim=(0, 2, 3)).sqrt(),
        }
        return self._finish(
            name,
            acc,
            step,
            {
                "source": "qk_recompute",
                "query_count": length,
                "sampled_query_count": indices.numel(),
                "key_count": key_length,
                "sampled": indices.numel() < length,
                "scale": factor,
                "key_heads": original_key_heads,
                "entropy_unit": "nats",
                "logit_statistics": "finite_unmasked_only",
            },
            extra,
        )

    @torch.no_grad()
    def observe_logits(self, name, logits, step, attn_mask=None, is_causal=False):
        """统计模型已算出的缩放后 logits，形状为 ``[..., H, L, S]``。

        H 为头数，L 为 query 数，S 为 key 数。应传入 dropout 前的 logits；
        其中可以已包含加性偏置或 mask，输入中的 -inf 统一视为结构性遮蔽。
        仅当 mask 尚未加到 logits 上时才传入 ``attn_mask``，避免重复施加。
        NaN 和 +inf 仍计为数值异常。此入口避免重算 QK，但需要模型显式暴露 logits。
        """
        if not self.is_log_step(step):
            return None
        if logits.ndim < 3 or not logits.is_floating_point():
            raise ValueError("logits must be floating point [..., heads, queries, keys]")
        if any(size == 0 for size in logits.shape):
            raise ValueError("logits must be nonempty")
        logits = logits.detach()
        heads, length, key_length = logits.shape[-3:]
        mask = self._validate_mask(attn_mask, logits.shape, logits.device)
        indices = self._query_indices(length, logits.device)
        acc = self._new_accumulator(heads, logits.device)
        for chunk in indices.split(self.query_chunk_size):
            values = logits.index_select(-2, chunk).float()
            # 约定输入中已有的 -inf 表示 Attention 遮蔽，不计为数值故障。
            original_allowed = ~torch.isneginf(values)
            values, allowed = self._apply_mask(values, mask, chunk, is_causal)
            self._accumulate(acc, values, allowed & original_allowed)
        return self._finish(
            name,
            acc,
            step,
            {
                "source": "provided_logits",
                "query_count": length,
                "sampled_query_count": indices.numel(),
                "key_count": key_length,
                "sampled": indices.numel() < length,
                "entropy_unit": "nats",
                "logit_statistics": "finite_unmasked_only",
            },
        )
