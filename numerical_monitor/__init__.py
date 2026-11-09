"""Training numerical monitoring with explicit sampling and phase boundaries."""

from .attention import AttentionMonitor
from .gradient_monitor import GradientMonitor
from .metrics import tensor_metrics
from .numerical_analyzer import MetricKey, NumericalAnalyzer

__version__ = "0.1.0"
__all__ = [
    "AttentionMonitor",
    "GradientMonitor",
    "MetricKey",
    "NumericalAnalyzer",
    "tensor_metrics",
]
