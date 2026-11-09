"""Explicit, sampled attention diagnostics without altering a model's forward path.

Q/K must have shape ``[..., heads, tokens, head_dim]``. These probes recompute
attention scores, so call them only in selected layers/steps. They do not disable
FlashAttention or request attention weights from the model.
"""

import math
from typing import Callable, Optional

import torch


class AttentionMonitor:
    """Collect per-head Q/K scale, finite logit maximum and attention entropy.

    Entropy uses natural logarithms and probabilities before attention dropout.
    Fully masked or numerically invalid query rows are excluded from its mean.
    Logit fault counts count eligible (unmasked) entries only. All returned
    statistics describe this rank's activations, not distributed-global values.

    Args:
        log_scalar_fn: Optional ``callback(tag, value, step)``.
        log_interval: Collect every this many optimizer steps.
        query_chunk_size: Maximum query rows in a temporary score matrix.
        max_query_samples: Cap deterministic, evenly spaced query samples. None
            computes all rows, in chunks; memory remains bounded but work is
            still quadratic in sequence length.
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
        return step % self.log_interval == 0

    def _query_indices(self, length, device):
        count = min(length, self.max_query_samples or length)
        if count == length:
            return torch.arange(length, device=device)
        return torch.linspace(0, length - 1, count, device=device).long()

    @staticmethod
    def _validate_mask(attn_mask, shape, device):
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
        allowed = torch.ones_like(logits, dtype=torch.bool)
        if mask is not None:
            chunk_mask = mask.index_select(-2, indices)
            if chunk_mask.dtype == torch.bool:
                allowed &= chunk_mask
            else:
                # -inf is a structural mask, while NaN/+inf are numeric faults.
                allowed &= ~torch.isneginf(chunk_mask)
                logits = logits + chunk_mask.to(dtype=logits.dtype)
        if is_causal:
            # Matches SDPA's upper-left causal alignment for nonsquare scores.
            keys = torch.arange(logits.shape[-1], device=logits.device)
            allowed &= keys[None, :] <= indices[:, None]
        return logits, allowed

    @staticmethod
    def _new_accumulator(heads, device):
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
        # Avoid softmax(-inf, ..., -inf); exclude these rows from statistics.
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
        count = acc.pop("valid_query_count")
        entropy_sum = acc.pop("entropy_sum")
        acc["entropy"] = torch.where(count > 0, entropy_sum / count.clamp_min(1), torch.nan)
        acc["valid_query_count"] = count
        acc["logit_max"] = torch.where(acc["logit_max"] == -torch.inf, torch.nan, acc["logit_max"])
        if extra:
            acc.update(extra)
        # Batch transfers by dtype: preserve exact integer counts without
        # requiring device-side FP64 (unsupported by some accelerators).
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
        """Explicitly recompute selected score rows from Q/K in FP32.

        Pass the actual Q/K after model-specific RoPE, normalization, and other
        transformations. ``attn_mask=True`` means allowed, following SDPA. Float
        masks are additive; negative infinity masks an entry. GQA requires an
        explicit ``enable_gqa=True`` and an integer Q-head/K-head ratio.
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
        # Explicit casts alone do not prevent an enclosing autocast context
        # from downcasting matmul again. Keep the diagnostic recompute in FP32.
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
        """Observe already computed, scaled logits ``[..., H, L, S]``.

        Logits should be before dropout. They may already include an additive
        bias/mask; negative infinity in supplied logits is treated as masked.
        Pass ``attn_mask`` only if it has not already been added to the logits.
        NaN and positive infinity remain numeric faults. This method avoids QK
        recomputation but requires the model to expose its logits explicitly.
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
            # Existing -inf values conventionally encode attention masking.
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
