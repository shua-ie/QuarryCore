"""A full-stack observability and monitoring system."""

from __future__ import annotations

from typing import Optional, Dict, Any

from .manager import ObservabilityManager
from .logging import configure_logging
from .metrics import MetricsManager, METRICS

__all__ = ["ObservabilityManager", "configure_logging", "MetricsManager", "METRICS"]

# Convenience functions for metrics
def increment(name: str, value: float = 1.0, labels: Optional[Dict[str, Any]] = None) -> None:
    """Increment a counter metric."""
    if name in METRICS:
        metric = METRICS[name]
        if labels is not None:
            metric.labels(**labels).inc(value)
        else:
            metric.inc(value)

def gauge(name: str, value: float, labels: Optional[Dict[str, Any]] = None) -> None:
    """Set a gauge metric."""
    if name in METRICS:
        metric = METRICS[name]
        if labels is not None:
            metric.labels(**labels).set(value)
        else:
            metric.set(value)

def histogram(name: str, value: float, labels: Optional[Dict[str, Any]] = None) -> None:
    """Observe a histogram metric."""
    if name in METRICS:
        metric = METRICS[name]
        if labels is not None:
            metric.labels(**labels).observe(value)
        else:
            metric.observe(value)

def export_prometheus() -> str:
    """Export metrics in Prometheus format."""
    from prometheus_client import generate_latest
    return generate_latest().decode('utf-8')

def get_all_metrics() -> Dict[str, Any]:
    """Get all current metric values."""
    return {name: metric._value.get() if hasattr(metric, '_value') else 0 for name, metric in METRICS.items()} 