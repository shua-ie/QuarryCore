"""
Central manager for the observability and monitoring system.
"""

from __future__ import annotations

import threading
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, AsyncContextManager, AsyncIterator, Dict
from uuid import UUID

import structlog

from quarrycore.protocols import ObservabilityProtocol
from quarrycore.web.main import run_web_server

from .logging import configure_logging
from .metrics import METRICS, MetricsManager

if TYPE_CHECKING:
    from quarrycore.config.config import MonitoringConfig
    from quarrycore.protocols import ErrorInfo, PerformanceMetrics


class ObservabilityManager(ObservabilityProtocol):
    """
    Orchestrates logging, metrics, and the web UI for the application.
    """

    def __init__(self, config: MonitoringConfig) -> None:
        self.config = config
        self.logger = structlog.get_logger(self.__class__.__name__)
        self.metrics_manager = MetricsManager(config)
        self._web_server_thread: threading.Thread | None = None
        self._is_running = False

    async def __aenter__(self) -> "ObservabilityManager":
        """Enter the async context manager."""
        if not self.config.enabled:
            return self

        if not self._is_running:
            # Configure logging
            configure_logging(self.config)

            # Start metrics manager (synchronous)
            self.metrics_manager.start()

            # Start web server if enabled
            if self.config.web_ui.enabled:
                self._web_server_thread = threading.Thread(
                    target=run_web_server,
                    args=(self.config.web_ui.host, self.config.web_ui.port),
                    daemon=True,
                )
                self._web_server_thread.start()

            self._is_running = True
            self.logger.info("Observability manager started")

        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit the async context manager."""
        if self._is_running:
            self.shutdown()

    def start_monitoring(self) -> AsyncContextManager[None]:
        """A context manager to start and stop all monitoring services."""
        return self._monitoring_context()

    @asynccontextmanager
    async def _monitoring_context(self) -> AsyncIterator[None]:
        """Internal context manager implementation."""
        if not self.config.enabled:
            yield
            return

        if self._is_running:
            yield
            return

        configure_logging(self.config)
        self.metrics_manager.start()

        if self.config.web_ui.enabled:
            self._web_server_thread = threading.Thread(
                target=run_web_server,
                args=(self.config.web_ui.host, self.config.web_ui.port),
                daemon=True,
            )
            self._web_server_thread.start()

        self._is_running = True
        self.logger.info("Observability manager started.")

        try:
            yield
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        """Shuts down all monitoring services."""
        if not self._is_running:
            return

        self.metrics_manager.stop()
        # The web server thread is a daemon, so it will exit with the main app.
        # For a more graceful shutdown, we'd need to use uvicorn's programmatic API.
        self.logger.info("Observability manager shut down.")
        self._is_running = False

    async def log_performance_metrics(
        self,
        component: str,
        metrics: PerformanceMetrics,
        correlation_id: UUID | None = None,
    ) -> None:
        METRICS["processing_duration_seconds"].labels(pipeline_stage=component).observe(
            metrics.total_duration_ms / 1000
        )  # type: ignore
        self.logger.info(
            "performance_metric",
            component=component,
            metrics=metrics,
            correlation_id=str(correlation_id) if correlation_id else None,
        )

    async def log_error(
        self,
        error: ErrorInfo,
        component: str,
        correlation_id: UUID | None = None,
    ) -> None:
        self.logger.error(
            "application_error",
            component=component,
            error_type=error.error_type,
            error_message=error.error_message,
            severity=error.severity.value,
            correlation_id=str(correlation_id) if correlation_id else None,
        )

    # --- Other protocol methods ---
    async def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health status."""
        return {
            "status": "healthy" if self._is_running else "stopped",
            "monitoring_enabled": self.config.enabled,
            "web_ui_enabled": self.config.web_ui.enabled,
            "metrics_collected": True,
            "uptime_seconds": 0,  # Would need to track start time
        }

    async def get_performance_report(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Generate performance analysis report."""
        return {
            "report_generated": True,
            "metrics_available": self._is_running,
            "time_range": "not_implemented",
        }

    async def export_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format."""
        if hasattr(self.metrics_manager, "registry"):
            from prometheus_client import generate_latest

            return generate_latest(self.metrics_manager.registry).decode("utf-8")
        return "# No metrics available\n"

    async def create_alert(self, *args: Any, **kwargs: Any) -> UUID:
        """Create monitoring alert with automatic actions."""
        from uuid import uuid4

        alert_id = uuid4()
        self.logger.info("Alert created", alert_id=str(alert_id))
        return alert_id
