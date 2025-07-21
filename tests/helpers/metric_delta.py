"""
Helpers for validating metric value changes during tests.

Provides context managers to ensure metrics are properly updated by the code under test.
"""

from contextlib import contextmanager
from typing import Any, Union


@contextmanager
def metric_delta(metric, expected_delta=1):
    """
    Context manager to validate metric value changes.

    Args:
        metric: Prometheus metric object
        expected_delta: Expected change in metric value

    Usage:
        with metric_delta(METRICS["some_counter"]):
            # Code that should increment counter by 1
            pass

        with metric_delta(METRICS["some_gauge"], 5):
            # Code that should increase gauge by 5
            pass
    """
    if hasattr(metric, "_value"):
        initial_value = metric._value.get()
    else:
        raise ValueError(f"Metric {metric} doesn't have a _value attribute")

    yield

    final_value = metric._value.get()
    actual_delta = final_value - initial_value

    if actual_delta != expected_delta:
        raise AssertionError(
            f"Expected metric to change by {expected_delta}, "
            f"but it changed by {actual_delta} "
            f"(from {initial_value} to {final_value})"
        )


@contextmanager
def metric_increases(metric):
    """
    Context manager to validate that a metric increases (any positive amount).

    Usage:
        with metric_increases(METRICS["request_count"]):
            # Code that should increase the counter
            pass
    """
    if hasattr(metric, "_value"):
        initial_value = metric._value.get()
    else:
        raise ValueError(f"Metric {metric} doesn't have a _value attribute")

    yield

    final_value = metric._value.get()
    actual_delta = final_value - initial_value

    if actual_delta <= 0:
        raise AssertionError(
            f"Expected metric to increase, but it changed by {actual_delta} (from {initial_value} to {final_value})"
        )


def get_histogram_count(histogram):
    """Get the current observation count for a histogram."""
    try:
        collected = list(histogram.collect())
        if collected:
            for sample in collected[0].samples:
                if sample.name.endswith("_count"):
                    return sample.value
        return 0.0
    except (AttributeError, TypeError, IndexError):
        return 0.0


@contextmanager
def histogram_observes(histogram, min_observations=1):
    """
    Context manager to validate histogram observations.

    Args:
        histogram: Prometheus histogram metric
        min_observations: Minimum number of observations expected

    Usage:
        with histogram_observes(METRICS["request_duration"]):
            # Code that should record at least one timing observation
            pass
    """
    initial_count = get_histogram_count(histogram)

    yield

    final_count = get_histogram_count(histogram)
    actual_observations = final_count - initial_count

    if actual_observations < min_observations:
        raise AssertionError(
            f"Expected at least {min_observations} histogram observations, "
            f"but got {actual_observations} "
            f"(count went from {initial_count} to {final_count})"
        )
