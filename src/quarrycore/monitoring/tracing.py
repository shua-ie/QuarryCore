"""
Distributed tracing for QuarryCore using OpenTelemetry.

Provides request correlation and performance bottleneck identification.
"""
from __future__ import annotations

import asyncio
import functools
from contextlib import asynccontextmanager
from typing import Any, Callable, Dict, Optional
from uuid import UUID

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Status, StatusCode

from quarrycore.protocols import create_correlation_id


class TracingManager:
    """
    Manages distributed tracing with OpenTelemetry.
    
    Features:
    - Request correlation across all components
    - Span creation for each processing step
    - Performance bottleneck identification
    - Cross-service trace correlation
    - Error tracking with stack traces
    """
    
    def __init__(
        self,
        service_name: str = "quarrycore",
        otlp_endpoint: Optional[str] = None,
        enabled: bool = True,
    ):
        self.service_name = service_name
        self.enabled = enabled
        self.tracer = None
        
        if enabled:
            # Set up OpenTelemetry
            resource = Resource.create({
                "service.name": service_name,
                "service.version": "1.0.0",
            })
            
            provider = TracerProvider(resource=resource)
            
            # Configure exporter
            if otlp_endpoint:
                exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
                processor = BatchSpanProcessor(exporter)
                provider.add_span_processor(processor)
            
            trace.set_tracer_provider(provider)
            self.tracer = trace.get_tracer(service_name)
    
    @asynccontextmanager
    async def trace_operation(
        self,
        operation_name: str,
        attributes: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[UUID] = None,
    ):
        """
        Trace an async operation.
        
        Usage:
            async with tracing.trace_operation("process_document", {"url": url}):
                await process_document(url)
        """
        if not self.enabled or not self.tracer:
            yield
            return
        
        # Create correlation ID if not provided
        if correlation_id is None:
            correlation_id = create_correlation_id()
        
        # Start span
        with self.tracer.start_as_current_span(operation_name) as span:
            # Set attributes
            span.set_attribute("correlation_id", str(correlation_id))
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, str(value))
            
            try:
                yield span
                span.set_status(Status(StatusCode.OK))
            except Exception as e:
                # Record exception
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise
    
    def trace_method(
        self,
        operation_name: Optional[str] = None,
        extract_attributes: Optional[Callable] = None,
    ):
        """
        Decorator for tracing methods.
        
        Usage:
            @trace_method("crawl_url")
            async def crawl_url(self, url: str):
                ...
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                # Determine operation name
                op_name = operation_name or f"{func.__module__}.{func.__name__}"
                
                # Extract attributes if function provided
                attributes = {}
                if extract_attributes:
                    attributes = extract_attributes(*args, **kwargs)
                
                # Trace the operation
                async with self.trace_operation(op_name, attributes):
                    return await func(*args, **kwargs)
            
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                # For sync methods, we can't use async context manager
                if not self.enabled or not self.tracer:
                    return func(*args, **kwargs)
                
                op_name = operation_name or f"{func.__module__}.{func.__name__}"
                
                with self.tracer.start_as_current_span(op_name) as span:
                    try:
                        result = func(*args, **kwargs)
                        span.set_status(Status(StatusCode.OK))
                        return result
                    except Exception as e:
                        span.record_exception(e)
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        raise
            
            # Return appropriate wrapper
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        
        return decorator
    
    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        """Add an event to the current span."""
        if not self.enabled:
            return
        
        current_span = trace.get_current_span()
        if current_span and current_span.is_recording():
            current_span.add_event(name, attributes=attributes or {})
    
    def set_attribute(self, key: str, value: Any) -> None:
        """Set an attribute on the current span."""
        if not self.enabled:
            return
        
        current_span = trace.get_current_span()
        if current_span and current_span.is_recording():
            current_span.set_attribute(key, str(value))
    
    def get_current_trace_id(self) -> Optional[str]:
        """Get the current trace ID."""
        if not self.enabled:
            return None
        
        current_span = trace.get_current_span()
        if current_span and current_span.is_recording():
            span_context = current_span.get_span_context()
            return format(span_context.trace_id, '032x')
        
        return None


# Global tracing instance
_tracing_manager: Optional[TracingManager] = None

def init_tracing(
    service_name: str = "quarrycore",
    otlp_endpoint: Optional[str] = None,
    enabled: bool = True
) -> TracingManager:
    """Initialize global tracing."""
    global _tracing_manager
    _tracing_manager = TracingManager(service_name, otlp_endpoint, enabled)
    return _tracing_manager

def trace_operation(
    operation_name: str,
    attributes: Optional[Dict[str, Any]] = None,
    correlation_id: Optional[UUID] = None,
):
    """Trace an operation using the global tracer."""
    global _tracing_manager
    if _tracing_manager is None:
        _tracing_manager = init_tracing()
    
    return _tracing_manager.trace_operation(operation_name, attributes, correlation_id) 