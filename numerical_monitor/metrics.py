"""监控组件共用的张量统计：有限值分布与 NaN/Inf 故障分别统计。"""

import math

import torch


@torch.no_grad()
def tensor_metrics(tensor):
    """计算有限元素的总体统计量，并独立返回 NaN、正负 Inf 数量。

    空张量或全部非有限时，分布指标为 NaN，有限元素的 L2 norm 为零。
    不修改输入或保留计算图。浮点统计与整数计数分别打包传回主机，
    减少逐指标同步，并避免整数计数因转成低精度浮点而丢失精度。
    """
    if hasattr(tensor, "to_local"):
        tensor = tensor.to_local()
    if tensor.is_complex():
        raise TypeError("Real-valued tensors are required")
    dtype = torch.float64 if tensor.dtype == torch.float64 else torch.float32
    # 低精度输入提升到 FP32，但保留 FP64，避免微小变化在统计前被舍入。
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
    # 先按最大绝对值缩放，再计算均值/标准差/RMS，避免大有限值平方溢出。
    # 非零次正规数使用其真实尺度；只有全零输入才用 1 避免除零。
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
    # 在主机侧恢复尺度，避免设备侧 L2 中间量溢出时连带污染仍可表示的 RMS。
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
    """读取脱离计算图的本地值；DTensor 仅取本地分片，不触发 all-gather。"""
    return tensor.to_local().detach() if hasattr(tensor, "to_local") else tensor.detach()
