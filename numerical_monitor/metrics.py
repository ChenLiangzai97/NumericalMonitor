"""Finite-aware tensor statistics shared by the monitors."""

import math

import torch


@torch.no_grad()
def tensor_metrics(tensor):
    """Return finite-only population statistics and separate fault counts.

    Empty/all-invalid tensors have NaN distribution statistics, zero finite L2
    norm and explicit counts. Does not change or retain the input/autograd graph.
    Distribution values and integer counts are transferred in two packed batches.
    """
    if hasattr(tensor, "to_local"):
        tensor = tensor.to_local()
    if tensor.is_complex():
        raise TypeError("Real-valued tensors are required")
    dtype = torch.float64 if tensor.dtype == torch.float64 else torch.float32
    x = tensor.detach().to(dtype).reshape(-1)
    finite = torch.isfinite(x)
    n = finite.sum()
    safe = torch.where(finite, x, 0.0)
    count = n.clamp_min(1)
    if x.numel():
        maximum = torch.where(finite, x, -torch.inf).amax()
        minimum = torch.where(finite, x, torch.inf).amin()
        absmax = safe.abs().amax()
    else:
        maximum = x.new_tensor(-torch.inf)
        minimum = x.new_tensor(torch.inf)
        absmax = x.new_zeros(())
    # Scale the sum of squares to avoid overflow from squaring large values.
    scale = torch.where(absmax > 0, absmax, torch.ones_like(absmax))
    normalized = safe / scale
    mean = normalized.sum() / count
    centered = torch.where(finite, normalized - mean, 0.0)
    std = (centered.square().sum() / count).sqrt()
    norm = normalized.square().sum().sqrt()
    values = (
        torch.stack(
            [
                maximum,
                minimum,
                mean,
                std,
                absmax,
                norm,
                norm / count.sqrt(),
                ((x == 0) & finite).sum() / count,
            ]
        )
        .cpu()
        .tolist()
    )
    nf, nan, pi, ni = (
        torch.stack(
            [
                n,
                torch.isnan(x).sum(),
                torch.isposinf(x).sum(),
                torch.isneginf(x).sum(),
            ]
        )
        .cpu()
        .tolist()
    )
    mx, mn, avg, std, ma, l2, rms, zr = values
    avg, std, l2, rms = (v * ma for v in (avg, std, l2, rms))
    if not nf:
        mx = mn = avg = std = rms = zr = math.nan
    return dict(
        numel=x.numel(),
        finite_count=int(nf),
        max=mx,
        min=mn,
        max_diff=mx - mn,
        mean=avg,
        std=std,
        max_abs=ma,
        l2_norm=l2,
        rms=rms,
        zero_ratio=zr,
        nan_count=int(nan),
        posinf_count=int(pi),
        neginf_count=int(ni),
        inf_count=int(pi + ni),
    )


def local_tensor(tensor):
    """Get a detached local tensor without materializing a distributed tensor."""
    return tensor.to_local().detach() if hasattr(tensor, "to_local") else tensor.detach()
