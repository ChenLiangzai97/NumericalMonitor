"""Run a tiny CPU training loop and export all three monitoring paths."""

import argparse
from pathlib import Path

import torch
from torch import nn

from numerical_monitor import AttentionMonitor, GradientMonitor, NumericalAnalyzer


class ResidualBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.ffn = nn.Sequential(nn.Linear(width, width), nn.GELU())

    def forward(self, x):
        return x + self.ffn(x)


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(8, 16)
        self.block = ResidualBlock(16)
        self.head = nn.Linear(16, 3)

    def forward(self, x):
        return self.head(self.block(self.encoder(x)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--output", default="numerical_logs/toy")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    torch.manual_seed(7)
    output = Path(args.output)
    model = ToyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scalar_count = 0

    def log_scalar(tag, value, step):
        nonlocal scalar_count
        scalar_count += 1

    analyzer = NumericalAnalyzer(
        model,
        log_scalar_fn=log_scalar,
        log_interval=2,
        save_dir=str(output / "forward"),
        scan_parameters=False,
        include_modules=[r"model\.(block|head)$"],
        residual_modules=[r"model\.block$"],
        output_head_modules=[r"model\.head$"],
    )
    monitor = GradientMonitor(
        model,
        optimizer,
        log_scalar_fn=log_scalar,
        save_dir=output / "training",
        grad_interval=1,
        param_interval=2,
        optimizer_interval=2,
        update_interval=2,
        output_heads=("head",),
        adam_denominator=True,
    )
    attention = AttentionMonitor(
        log_scalar, log_interval=2, query_chunk_size=2, max_query_samples=3
    )
    analyzer.enable()
    try:
        for step in range(args.steps):
            analyzer.begin_step(step)
            optimizer.zero_grad(set_to_none=True)
            x, target = torch.randn(4, 8), torch.randn(4, 3)
            prediction = model(x)
            per_sample_loss = (prediction - target).square().mean(dim=-1)
            monitor.record_loss(per_sample_loss, step)
            loss = per_sample_loss.mean()
            loss.backward()
            monitor.capture_gradients(step)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            monitor.record_clipping(step)
            monitor.before_optimizer_step(step)
            optimizer.step()
            monitor.after_optimizer_step(step)
            # Independent probe input illustrates the model-internal Q/K API;
            # these tensors are not claimed to come from ToyModel's FFN.
            q, k = torch.randn(1, 2, 5, 4), torch.randn(1, 2, 5, 4)
            attention.observe("example_attention", q, k, step, is_causal=True)
            analyzer.step_forward(step)
            monitor.flush()
            print(f"step={step} loss={loss.detach().item():.6f}")
    finally:
        monitor.flush()
        analyzer.disable()
    print(f"Exported metrics to {output.resolve()} ({scalar_count} scalar events)")


if __name__ == "__main__":
    main()
