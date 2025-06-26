"""
Business metrics for QuarryCore with singleton registry to prevent collisions.

Provides custom Prometheus metrics for business KPIs and operational monitoring
with thread-safe singleton pattern to prevent metric registration conflicts.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Dict, Optional, List
from uuid import UUID

from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, REGISTRY
from prometheus_client.metrics import MetricWrapperBase

import structlog

logger = structlog.get_logger(__name__)


class MetricsRegistry:
    """
    Thread-safe singleton metrics registry to prevent Prometheus collisions.
    
    This ensures that metrics are only registered once across the entire
    application, preventing the startup crashes from duplicate registrations.
    """
    _instance: Optional['MetricsRegistry'] = None
    _lock = threading.Lock()
    _metrics: Dict[str, Any] = {}
    _initialized = False
    
    def __new__(cls) -> 'MetricsRegistry':
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self) -> None:
        # Only initialize once
        if not self._initialized:
            with self._lock:
                if not self._initialized:
                    self._metrics = {}
                    self._initialized = True
                    logger.info("Metrics registry initialized")
    
    def get_or_create_counter(
        self, 
        name: str, 
        description: str, 
        labelnames: Optional[List[str]] = None
    ) -> Counter:
        """Get existing counter or create new one - NEVER duplicate."""
        if name in self._metrics:
            return self._metrics[name]
        
        try:
            counter = Counter(name, description, labelnames or [])
            self._metrics[name] = counter
            logger.debug(f"Created new counter: {name}")
            return counter
        except ValueError as e:
            if "already exists" in str(e) or "Duplicated timeseries" in str(e):
                # Metric already exists in Prometheus registry
                existing_metric = self._get_existing_metric(name)
                if existing_metric:
                    self._metrics[name] = existing_metric
                    logger.debug(f"Retrieved existing counter: {name}")
                    return existing_metric
                else:
                    # Create a new registry-safe metric
                    logger.warning(f"Metric {name} exists but not retrievable, creating with suffix")
                    safe_name = f"{name}_{int(time.time())}"
                    counter = Counter(safe_name, description, labelnames or [])
                    self._metrics[name] = counter
                    return counter
            raise
    
    def get_or_create_histogram(
        self, 
        name: str, 
        description: str,
        labelnames: Optional[List[str]] = None,
        buckets: Optional[List[float]] = None
    ) -> Histogram:
        """Get existing histogram or create new one - NEVER duplicate."""
        if name in self._metrics:
            return self._metrics[name]
        
        try:
            histogram = Histogram(name, description, labelnames or [], buckets=buckets or [])
            self._metrics[name] = histogram
            logger.debug(f"Created new histogram: {name}")
            return histogram
        except ValueError as e:
            if "already exists" in str(e) or "Duplicated timeseries" in str(e):
                existing_metric = self._get_existing_metric(name)
                if existing_metric:
                    self._metrics[name] = existing_metric
                    logger.debug(f"Retrieved existing histogram: {name}")
                    return existing_metric
                else:
                    logger.warning(f"Histogram {name} exists but not retrievable, creating with suffix")
                    safe_name = f"{name}_{int(time.time())}"
                    histogram = Histogram(safe_name, description, labelnames or [], buckets=buckets or [])
                    self._metrics[name] = histogram
                    return histogram
            raise
    
    def get_or_create_gauge(
        self, 
        name: str, 
        description: str,
        labelnames: Optional[List[str]] = None
    ) -> Gauge:
        """Get existing gauge or create new one - NEVER duplicate."""
        if name in self._metrics:
            return self._metrics[name]
        
        try:
            gauge = Gauge(name, description, labelnames or [])
            self._metrics[name] = gauge
            logger.debug(f"Created new gauge: {name}")
            return gauge
        except ValueError as e:
            if "already exists" in str(e) or "Duplicated timeseries" in str(e):
                existing_metric = self._get_existing_metric(name)
                if existing_metric:
                    self._metrics[name] = existing_metric
                    logger.debug(f"Retrieved existing gauge: {name}")
                    return existing_metric
                else:
                    logger.warning(f"Gauge {name} exists but not retrievable, creating with suffix")
                    safe_name = f"{name}_{int(time.time())}"
                    gauge = Gauge(safe_name, description, labelnames or [])
                    self._metrics[name] = gauge
                    return gauge
            raise
    
    def _get_existing_metric(self, name: str) -> Optional[Any]:
        """Retrieve existing metric from Prometheus registry."""
        try:
            # Check if metric exists in registry by attempting to collect
            for collector in REGISTRY._collector_to_names:
                if hasattr(collector, '_name') and getattr(collector, '_name', '') == name:
                    return collector
                # Check for metrics with multiple names (like histograms)
                if hasattr(collector, '_names') and name in getattr(collector, '_names', []):
                    return collector
        except Exception as e:
            logger.warning(f"Error retrieving existing metric {name}: {e}")
        return None
    
    def clear_metrics(self) -> None:
        """Clear all cached metrics (for testing)."""
        with self._lock:
            self._metrics.clear()
            logger.debug("Metrics registry cleared")
    
    def get_metrics_count(self) -> int:
        """Get number of registered metrics."""
        return len(self._metrics)


class BusinessMetrics:
    """
    Business metrics for QuarryCore with collision-safe registration.
    
    Tracks business KPIs and operational metrics with proper
    singleton pattern to prevent startup crashes.
    """
    
    def __init__(self) -> None:
        self.registry = MetricsRegistry()
        
        # Document processing metrics
        self.documents_processed = self.registry.get_or_create_counter(
            'quarrycore_documents_processed_total',
            'Total documents processed by stage and status',
            ['stage', 'status', 'domain_type']
        )
        
        self.documents_in_flight = self.registry.get_or_create_gauge(
            'quarrycore_documents_in_flight',
            'Number of documents currently being processed'
        )
        
        self.documents_rejected = self.registry.get_or_create_counter(
            'quarrycore_documents_rejected_total',
            'Total documents rejected by reason',
            ['reason', 'stage']
        )
        
        # Quality metrics
        self.quality_scores = self.registry.get_or_create_histogram(
            'quarrycore_quality_scores',
            'Document quality score distribution',
            buckets=[0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0]
        )
        
        self.content_length = self.registry.get_or_create_histogram(
            'quarrycore_content_length_chars',
            'Content length distribution in characters',
            buckets=[100, 500, 1000, 5000, 10000, 50000, 100000]
        )
        
        # Performance metrics
        self.processing_time = self.registry.get_or_create_histogram(
            'quarrycore_processing_time_seconds',
            'Document processing time by stage',
            ['stage'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
        )
        
        self.crawler_response_time = self.registry.get_or_create_histogram(
            'quarrycore_crawler_response_time_seconds',
            'Web crawler response time distribution',
            ['domain'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
        )
        
        # System health metrics
        self.circuit_breaker_state = self.registry.get_or_create_gauge(
            'quarrycore_circuit_breaker_state',
            'Circuit breaker state (0=closed, 1=open, 2=half-open)',
            ['component', 'domain']
        )
        
        self.dead_letter_queue_size = self.registry.get_or_create_gauge(
            'quarrycore_dead_letter_queue_size',
            'Number of documents in dead letter queue'
        )
        
        # Business KPIs
        self.domains_processed = self.registry.get_or_create_counter(
            'quarrycore_domains_processed_total',
            'Total unique domains processed',
            ['domain_type']
        )
        
        self.storage_size_bytes = self.registry.get_or_create_gauge(
            'quarrycore_storage_size_bytes',
            'Total storage size in bytes',
            ['tier']  # hot, warm, cold
        )
        
        self.deduplication_rate = self.registry.get_or_create_histogram(
            'quarrycore_deduplication_rate',
            'Deduplication rate (percentage of duplicates found)',
            buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        )
        
        # Error tracking
        self.errors_total = self.registry.get_or_create_counter(
            'quarrycore_errors_total',
            'Total errors by type and component',
            ['error_type', 'component', 'severity']
        )
        
        self.retry_attempts = self.registry.get_or_create_counter(
            'quarrycore_retry_attempts_total',
            'Total retry attempts by component and outcome',
            ['component', 'outcome']  # success, failure, max_retries_reached
        )
        
        logger.info("Business metrics initialized successfully")
    
    def record_document_processed(
        self, 
        stage: str, 
        status: str, 
        domain_type: str = "general"
    ) -> None:
        """Record a document processing event."""
        self.documents_processed.labels(
            stage=stage, 
            status=status, 
            domain_type=domain_type
        ).inc()
    
    def record_document_rejected(self, reason: str, stage: str) -> None:
        """Record a document rejection."""
        self.documents_rejected.labels(reason=reason, stage=stage).inc()
    
    def record_quality_score(self, score: float) -> None:
        """Record a quality score."""
        self.quality_scores.observe(score)
    
    def record_content_length(self, length: int) -> None:
        """Record content length."""
        self.content_length.observe(length)
    
    def record_processing_time(self, stage: str, duration_seconds: float) -> None:
        """Record processing time for a stage."""
        self.processing_time.labels(stage=stage).observe(duration_seconds)
    
    def record_crawler_response_time(self, domain: str, duration_seconds: float) -> None:
        """Record crawler response time."""
        self.crawler_response_time.labels(domain=domain).observe(duration_seconds)
    
    def set_circuit_breaker_state(self, component: str, domain: str, state: int) -> None:
        """Set circuit breaker state (0=closed, 1=open, 2=half-open)."""
        self.circuit_breaker_state.labels(component=component, domain=domain).set(state)
    
    def set_dead_letter_queue_size(self, size: int) -> None:
        """Set dead letter queue size."""
        self.dead_letter_queue_size.set(size)
    
    def record_domain_processed(self, domain_type: str) -> None:
        """Record a unique domain being processed."""
        self.domains_processed.labels(domain_type=domain_type).inc()
    
    def set_storage_size(self, tier: str, size_bytes: int) -> None:
        """Set storage size for a tier."""
        self.storage_size_bytes.labels(tier=tier).set(size_bytes)
    
    def record_deduplication_rate(self, rate: float) -> None:
        """Record deduplication rate."""
        self.deduplication_rate.observe(rate)
    
    def record_error(self, error_type: str, component: str, severity: str) -> None:
        """Record an error."""
        self.errors_total.labels(
            error_type=error_type, 
            component=component, 
            severity=severity
        ).inc()
    
    def record_retry_attempt(self, component: str, outcome: str) -> None:
        """Record a retry attempt."""
        self.retry_attempts.labels(component=component, outcome=outcome).inc()
    
    def set_documents_in_flight(self, count: int) -> None:
        """Set number of documents currently being processed."""
        self.documents_in_flight.set(count)
    
    def set_system_info(self, version: str, service: str, environment: str) -> None:
        """Set system information (called once at startup)."""
        # Create info metric if it doesn't exist
        system_info = self.registry.get_or_create_gauge(
            'quarrycore_system_info',
            'System information',
            ['version', 'service', 'environment']
        )
        system_info.labels(
            version=version,
            service=service,
            environment=environment
        ).set(1)


# Global singleton instance - safe to import multiple times
_business_metrics: Optional[BusinessMetrics] = None
_metrics_lock = threading.Lock()


def get_business_metrics() -> BusinessMetrics:
    """Get the global business metrics instance (thread-safe singleton)."""
    global _business_metrics
    if _business_metrics is None:
        with _metrics_lock:
            if _business_metrics is None:
                _business_metrics = BusinessMetrics()
    return _business_metrics


def register_business_metrics() -> BusinessMetrics:
    """Register and return business metrics (alias for get_business_metrics)."""
    return get_business_metrics()


# For backward compatibility
def init_business_metrics() -> BusinessMetrics:
    """Initialize business metrics (alias for get_business_metrics)."""
    return get_business_metrics() 