"""
Defines and manages Prometheus metrics for the application.
"""

from __future__ import annotations

import os
import threading
import time
from typing import TYPE_CHECKING, Any, Dict, Optional

import psutil
from prometheus_client import REGISTRY as _PROM_REGISTRY
from prometheus_client import Counter as _OrigCounter
from prometheus_client import Gauge as _OrigGauge
from prometheus_client import Histogram as _OrigHistogram
from prometheus_client import start_http_server

if TYPE_CHECKING:
    from quarrycore.config.config import MonitoringConfig

# Third-party imports with graceful fallbacks
try:
    import pynvml  # type: ignore[import-not-found]

    HAS_PYNVML = True
except ImportError:
    pynvml = None
    HAS_PYNVML = False

# --- Metric Definitions ---

# Enable metrics in test mode for proper testing
_TEST_MODE = os.environ.get("QUARRY_TEST_MODE", "0") == "1"

# ---------------------------------------------------------------------------
# Duplicate-safe Prometheus metric wrappers
# ---------------------------------------------------------------------------
# The wrappers are *defined before* any metric creation so that the subsequent
# module-level `_create_metrics()` invocation can freely use the conventional
# Counter / Gauge / Histogram symbols without risking duplicate registration
# errors when the module is imported multiple times during the test suite.


def _duplicate_safe_factory(metric_cls):
    """Return a factory that reuses an existing collector if already present."""

    def _factory(name: str, documentation: str, *args, **kwargs):  # type: ignore[override]
        existing = _PROM_REGISTRY._names_to_collectors.get(name)
        if existing is not None:
            return existing  # type: ignore[return-value]

        try:
            return metric_cls(name, documentation, *args, **kwargs)  # type: ignore[call-arg]
        except ValueError:
            # Registration lost the race – fall back to the now-existing collector.
            return _PROM_REGISTRY._names_to_collectors[name]  # type: ignore[return-value]

    return _factory


# Public aliases for general use.
Counter = _duplicate_safe_factory(_OrigCounter)  # type: ignore[assignment]
Gauge = _duplicate_safe_factory(_OrigGauge)  # type: ignore[assignment]
Histogram = _duplicate_safe_factory(_OrigHistogram)  # type: ignore[assignment]

# Using a 'quarrycore' prefix for all custom metrics
METRICS: Dict[str, Any] = {}


def _create_metrics() -> Dict[str, Any]:
    """Create metrics with duplicate handling."""

    # In test mode, always create fresh metrics
    if _TEST_MODE:
        return {
            "documents_processed": Counter(
                "quarrycore_documents_processed_total",
                "Total number of documents processed by the pipeline",
                ["pipeline_stage"],
            ),
            "documents_in_flight": Gauge(
                "quarrycore_documents_in_flight",
                "Number of documents currently being processed",
            ),
            "processing_duration_seconds": Histogram(
                "quarrycore_processing_duration_seconds",
                "Time taken to process a document through a pipeline stage",
                ["pipeline_stage"],
            ),
            "quality_score": Histogram(
                "quarrycore_quality_score",
                "Distribution of document quality scores",
                ["domain"],
            ),
            "cpu_usage_percent": Gauge(
                "quarrycore_cpu_usage_percent",
                "Current CPU utilization of the system",
            ),
            "memory_usage_percent": Gauge(
                "quarrycore_memory_usage_percent",
                "Current memory utilization of the system",
            ),
            "gpu_usage_percent": Gauge(
                "quarrycore_gpu_usage_percent",
                "Current GPU utilization",
                ["gpu_id"],
            ),
            "gpu_memory_percent": Gauge(
                "quarrycore_gpu_memory_percent",
                "Current GPU memory utilization",
                ["gpu_id"],
            ),
            "gpu_temperature_celsius": Gauge(
                "quarrycore_gpu_temperature_celsius",
                "Current GPU temperature",
                ["gpu_id"],
            ),
            "resource_efficiency": Gauge(
                "quarrycore_resource_efficiency",
                "Resource utilization efficiency score",
            ),
            "system_load": Gauge(
                "quarrycore_system_load",
                "System load average",
            ),
            # Crawler-specific metrics
            "crawler_fetch_latency_seconds": Histogram(
                "quarrycore_crawler_fetch_latency_seconds",
                "Time taken to fetch a URL including retries",
                buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
            ),
            "crawler_responses_total": Counter(
                "quarrycore_crawler_responses_total", "Total number of HTTP responses by status class", ["status_class"]
            ),
            "crawler_in_flight_requests": Gauge(
                "quarrycore_crawler_in_flight_requests",
                "Number of HTTP requests currently in flight",
            ),
            "crawler_domain_backoff_total": Counter(
                "quarrycore_crawler_domain_backoff_total",
                "Total number of domains that entered backoff/cooldown",
            ),
            # Quality assessment metrics
            "quality_reject_total": Counter(
                "quarrycore_quality_reject_total",
                "Total number of documents rejected due to low quality",
            ),
            "quality_scorer_latency": Histogram(
                "quarrycore_quality_scorer_latency_seconds",
                "Time taken by individual quality scorers",
                ["scorer"],
                buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
            ),
            "quality_scorer_errors": Counter(
                "quarrycore_quality_scorer_errors_total",
                "Total number of errors in quality scorers",
                ["scorer"],
            ),
        }

    # Check if metrics are already registered
    for collector in list(_PROM_REGISTRY._collector_to_names.keys()):
        collector_names = _PROM_REGISTRY._collector_to_names.get(collector, set())
        if any(name.startswith("quarrycore_") for name in collector_names):
            # Metrics already exist, return empty dict
            return {}

    try:
        return {
            "documents_processed": Counter(
                "quarrycore_documents_processed_total",
                "Total number of documents processed by the pipeline",
                ["pipeline_stage"],
            ),
            "documents_in_flight": Gauge(
                "quarrycore_documents_in_flight",
                "Number of documents currently being processed",
            ),
            "processing_duration_seconds": Histogram(
                "quarrycore_processing_duration_seconds",
                "Time taken to process a document through a pipeline stage",
                ["pipeline_stage"],
            ),
            "quality_score": Histogram(
                "quarrycore_quality_score",
                "Distribution of document quality scores",
                ["domain"],
            ),
            "cpu_usage_percent": Gauge(
                "quarrycore_cpu_usage_percent",
                "Current CPU utilization of the system",
            ),
            "memory_usage_percent": Gauge(
                "quarrycore_memory_usage_percent",
                "Current memory utilization of the system",
            ),
            "gpu_usage_percent": Gauge(
                "quarrycore_gpu_usage_percent",
                "Current GPU utilization",
                ["gpu_id"],
            ),
            "gpu_memory_percent": Gauge(
                "quarrycore_gpu_memory_percent",
                "Current GPU memory utilization",
                ["gpu_id"],
            ),
            "gpu_temperature_celsius": Gauge(
                "quarrycore_gpu_temperature_celsius",
                "Current GPU temperature",
                ["gpu_id"],
            ),
            "resource_efficiency": Gauge(
                "quarrycore_resource_efficiency",
                "Resource utilization efficiency score",
            ),
            "system_load": Gauge(
                "quarrycore_system_load",
                "System load average",
            ),
            # Crawler-specific metrics
            "crawler_fetch_latency_seconds": Histogram(
                "quarrycore_crawler_fetch_latency_seconds",
                "Time taken to fetch a URL including retries",
                buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
            ),
            "crawler_responses_total": Counter(
                "quarrycore_crawler_responses_total", "Total number of HTTP responses by status class", ["status_class"]
            ),
            "crawler_in_flight_requests": Gauge(
                "quarrycore_crawler_in_flight_requests",
                "Number of HTTP requests currently in flight",
            ),
            "crawler_domain_backoff_total": Counter(
                "quarrycore_crawler_domain_backoff_total",
                "Total number of domains that entered backoff/cooldown",
            ),
            # Quality assessment metrics
            "quality_reject_total": Counter(
                "quarrycore_quality_reject_total",
                "Total number of documents rejected due to low quality",
            ),
            "quality_scorer_latency": Histogram(
                "quarrycore_quality_scorer_latency_seconds",
                "Time taken by individual quality scorers",
                ["scorer"],
                buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
            ),
            "quality_scorer_errors": Counter(
                "quarrycore_quality_scorer_errors_total",
                "Total number of errors in quality scorers",
                ["scorer"],
            ),
        }
    except Exception as e:
        print(f"Error creating metrics: {e}")
        return {}


# Always create metrics
METRICS = _create_metrics()


class GpuMonitor(threading.Thread):
    """A thread that periodically collects GPU metrics."""

    def __init__(self, interval: int = 5) -> None:
        super().__init__(daemon=True)
        self.interval = interval
        self._stop_event = threading.Event()
        self.pynvml: Optional[Any] = None
        self.handle_count = 0

        if HAS_PYNVML and pynvml:
            try:
                pynvml.nvmlInit()
                self.pynvml = pynvml
                self.handle_count = pynvml.nvmlDeviceGetCount()
                print(f"GPU Monitor: Found {self.handle_count} GPUs.")
            except Exception as e:
                self.pynvml = None
                self.handle_count = 0
                print(f"GPU Monitor: Failed to initialize pynvml: {e}")
        else:
            self.pynvml = None
            self.handle_count = 0
            print("GPU Monitor: pynvml not available. No GPU metrics will be collected.")

    def run(self) -> None:
        """Periodically query and update GPU metrics."""
        if not self.pynvml:
            return

        while not self._stop_event.is_set():
            try:
                for i in range(self.handle_count):
                    handle = self.pynvml.nvmlDeviceGetHandleByIndex(i)
                    util = self.pynvml.nvmlDeviceGetUtilizationRates(handle)
                    mem = self.pynvml.nvmlDeviceGetMemoryInfo(handle)

                    # Update GPU utilization metrics
                    gpu_usage_metric = METRICS.get("gpu_usage_percent")
                    if gpu_usage_metric and hasattr(gpu_usage_metric, "labels"):
                        gpu_usage_metric.labels(gpu_id=i).set(util.gpu)

                    gpu_memory_metric = METRICS.get("gpu_memory_percent")
                    if gpu_memory_metric and hasattr(gpu_memory_metric, "labels"):
                        gpu_memory_metric.labels(gpu_id=i).set(100 * mem.used / mem.total)

                    gpu_temp_metric = METRICS.get("gpu_temperature_celsius")
                    if gpu_temp_metric and hasattr(gpu_temp_metric, "labels"):
                        gpu_temp_metric.labels(gpu_id=i).set(
                            self.pynvml.nvmlDeviceGetTemperature(handle, self.pynvml.NVML_TEMPERATURE_GPU)
                        )
            except Exception as e:
                print(f"GPU Monitor: Error collecting metrics: {e}")

            # Use event wait with timeout instead of blocking sleep for cooperative shutdown
            self._stop_event.wait(timeout=self.interval)

    def stop(self) -> None:
        """Stop the GPU monitoring thread."""
        self._stop_event.set()
        if self.pynvml and HAS_PYNVML:
            try:
                self.pynvml.nvmlShutdown()
            except Exception as e:
                print(f"GPU Monitor: Error during shutdown: {e}")


class MetricsManager:
    """Manages the lifecycle of metrics collection and exporting."""

    def __init__(self, config: MonitoringConfig) -> None:
        self.config = config
        self._gpu_monitor = GpuMonitor()

        # Initialize tracking variables for resource efficiency calculation
        self.last_cpu_usage: Optional[float] = None
        self.last_memory_usage: Optional[float] = None
        self.last_update_time: Optional[float] = None

    def start(self) -> None:
        """Starts the Prometheus server and GPU monitor thread."""
        if self.config.prometheus_port:
            print(f"Starting Prometheus metrics server on port {self.config.prometheus_port}")
            start_http_server(self.config.prometheus_port)

        self._gpu_monitor.start()

    def stop(self) -> None:
        """Stops background monitoring threads."""
        self._gpu_monitor.stop()

    def update_system_metrics(self) -> None:
        """Updates CPU and Memory metrics."""
        try:
            cpu_usage = psutil.cpu_percent()
            memory_usage = psutil.virtual_memory().percent

            cpu_metric = METRICS.get("cpu_usage_percent")
            if cpu_metric and hasattr(cpu_metric, "set"):
                cpu_metric.set(cpu_usage)

            memory_metric = METRICS.get("memory_usage_percent")
            if memory_metric and hasattr(memory_metric, "set"):
                memory_metric.set(memory_usage)

            # Update resource efficiency
            self._update_resource_efficiency(cpu_usage, memory_usage)

        except Exception as e:
            print(f"Error updating system metrics: {e}")

    def _calculate_efficiency(self, current_cpu: float, current_memory: float) -> float:
        """Calculate resource efficiency score based on CPU and memory usage."""
        # Simple efficiency calculation: lower usage = higher efficiency
        # This can be customized based on specific requirements
        cpu_efficiency = max(0.0, 100.0 - current_cpu) / 100.0
        memory_efficiency = max(0.0, 100.0 - current_memory) / 100.0

        # Weighted average (CPU weighted more heavily)
        efficiency = (cpu_efficiency * 0.7) + (memory_efficiency * 0.3)
        return min(1.0, max(0.0, efficiency))

    def _update_resource_efficiency(self, current_cpu: float, current_memory: float) -> None:
        """Update resource efficiency metrics."""
        try:
            # Calculate efficiency
            efficiency = self._calculate_efficiency(current_cpu, current_memory)

            # Update efficiency metric
            efficiency_metric = METRICS.get("resource_efficiency")
            if efficiency_metric and hasattr(efficiency_metric, "set"):
                efficiency_metric.set(efficiency)

            # Update system load
            try:
                load_avg = psutil.getloadavg()
                system_load_metric = METRICS.get("system_load")
                if system_load_metric and hasattr(system_load_metric, "set"):
                    system_load_metric.set(load_avg[0] if load_avg else 0.0)
            except (AttributeError, OSError):
                # getloadavg not available on all platforms
                pass

            # Update tracking variables
            self.last_cpu_usage = current_cpu
            self.last_memory_usage = current_memory
            self.last_update_time = time.time()

        except Exception as e:
            print(f"Error updating resource efficiency: {e}")

    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current metric values as a dictionary."""
        metrics_data: Dict[str, Any] = {}

        try:
            # System metrics
            metrics_data["cpu_usage"] = psutil.cpu_percent()
            metrics_data["memory_usage"] = psutil.virtual_memory().percent

            # Load average (if available)
            try:
                load_avg = psutil.getloadavg()
                metrics_data["load_average"] = load_avg[0] if load_avg else 0.0
            except (AttributeError, OSError):
                metrics_data["load_average"] = 0.0

            # GPU metrics (if available)
            if self._gpu_monitor.pynvml and self._gpu_monitor.handle_count > 0:
                gpu_metrics = []
                for i in range(self._gpu_monitor.handle_count):
                    try:
                        handle = self._gpu_monitor.pynvml.nvmlDeviceGetHandleByIndex(i)
                        util = self._gpu_monitor.pynvml.nvmlDeviceGetUtilizationRates(handle)
                        mem = self._gpu_monitor.pynvml.nvmlDeviceGetMemoryInfo(handle)

                        gpu_metrics.append(
                            {
                                "gpu_id": i,
                                "usage_percent": util.gpu,
                                "memory_percent": 100 * mem.used / mem.total,
                                "temperature": self._gpu_monitor.pynvml.nvmlDeviceGetTemperature(
                                    handle,
                                    self._gpu_monitor.pynvml.NVML_TEMPERATURE_GPU,
                                ),
                            }
                        )
                    except Exception as e:
                        print(f"Error getting GPU {i} metrics: {e}")

                metrics_data["gpu_metrics"] = gpu_metrics

        except Exception as e:
            print(f"Error collecting current metrics: {e}")

        return metrics_data


# Global metrics manager instance (to be initialized with config)
_metrics_manager: Optional[MetricsManager] = None


def get_metrics_manager() -> Optional[MetricsManager]:
    """Get the global metrics manager instance."""
    return _metrics_manager


def set_metrics_manager(manager: MetricsManager) -> None:
    """Set the global metrics manager instance."""
    global _metrics_manager
    _metrics_manager = manager


# A global instance for easy access from other modules
# In a real DI system, this would be injected.
# metrics_manager = MetricsManager(some_config)
# METRICS["documents_processed"].labels(pipeline_stage="crawl").inc()
