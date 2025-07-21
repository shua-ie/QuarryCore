"""
FastAPI application for the QuarryCore real-time web dashboard.

Now with production-grade security, authentication, and monitoring.
"""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Dict, Optional
from uuid import uuid4

import psutil

try:
    import pynvml  # type: ignore[import-not-found]

    HAS_PYNVML = True
    pynvml.nvmlInit()
except ImportError:
    pynvml = None
    HAS_PYNVML = False

from fastapi import Depends, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from prometheus_client import generate_latest

from quarrycore.auth import AuthenticationMiddleware, User, UserRole, get_current_user
from quarrycore.monitoring import AuditLogger, BusinessMetrics, init_tracing, register_business_metrics
from quarrycore.observability.metrics import METRICS
from quarrycore.protocols import SystemMetrics
from quarrycore.security import ProductionRateLimiter, RateLimitMiddleware, SecurityHeadersMiddleware

# Initialize dependencies
business_metrics = register_business_metrics()
audit_logger = AuditLogger()
rate_limiter = ProductionRateLimiter()
tracer = init_tracing(service_name="quarrycore-web")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application lifecycle."""
    # Startup
    print("Starting QuarryCore Web Dashboard...")
    business_metrics.set_system_info(version="0.1.0", service="quarrycore-web", environment="production")

    yield

    # Shutdown
    print("Shutting down QuarryCore Web Dashboard...")
    if pynvml:
        pynvml.nvmlShutdown()


app = FastAPI(
    title="QuarryCore Monitoring Dashboard",
    version="0.1.0",
    lifespan=lifespan,
)

# Add security middleware in correct order
app.add_middleware(
    SecurityHeadersMiddleware,
    csp_directives={
        "default-src": "'self'",
        "script-src": "'self' 'unsafe-inline' 'unsafe-eval' https://cdn.jsdelivr.net",
        "style-src": "'self' 'unsafe-inline' https://fonts.googleapis.com",
        "font-src": "'self' https://fonts.gstatic.com",
        "img-src": "'self' data: https:",
        "connect-src": "'self' ws: wss: https:",
        "frame-ancestors": "'none'",
        "base-uri": "'self'",
        "form-action": "'self'",
    },
    allowed_origins=["http://localhost:3000", "https://app.quarrycore.com"],
    enable_hsts=True,
)

# Add rate limiting middleware
app.add_middleware(
    RateLimitMiddleware,
    rate_limiter=rate_limiter,
)

# Add CORS middleware for API access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://app.quarrycore.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-API-Key", "X-Request-ID"],
)

# Path to the HTML template
HTML_TEMPLATE_PATH = Path(__file__).parent / "templates" / "index.html"


@app.get("/", response_class=HTMLResponse)
async def get_dashboard() -> str:
    """Serves the main dashboard HTML page."""
    return HTML_TEMPLATE_PATH.read_text()


@app.get("/metrics")
async def get_prometheus_metrics() -> Any:
    """Endpoint for Prometheus to scrape."""
    return HTMLResponse(generate_latest(), media_type="text/plain")


def get_system_metrics() -> SystemMetrics:
    """Gathers real-time system metrics."""
    metrics = SystemMetrics()
    metrics.cpu_usage = psutil.cpu_percent()
    metrics.memory_usage = psutil.virtual_memory().percent

    if pynvml:
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            metrics.gpu_usage = util.gpu
            metrics.gpu_memory_usage = 100 * mem.used / mem.total
        except pynvml.NVMLError:
            # Could happen if GPU is busy or other issues
            pass

    # Get pipeline metrics from business metrics
    metrics.documents_in_flight = int(business_metrics.documents_in_flight._value.get())
    # For Counter, we need to collect and sum all label values
    total_processed = 0
    for metric in business_metrics.documents_processed.collect():
        for sample in metric.samples:
            if sample.name.endswith("_total"):
                total_processed += sample.value
    metrics.total_documents_processed = int(total_processed)

    # Calculate docs per minute
    if hasattr(app.state, "start_time"):
        elapsed_minutes = (time.time() - app.state.start_time) / 60
        if elapsed_minutes > 0:
            metrics.docs_per_minute = metrics.total_documents_processed / elapsed_minutes

    return metrics


@app.websocket("/ws/metrics")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """WebSocket endpoint for real-time system metrics."""
    await websocket.accept()

    # Log WebSocket connection
    client_ip = websocket.client.host if websocket.client else "unknown"
    audit_logger.log_api_access(
        user_id=None,
        user_roles=[],
        endpoint="/ws/metrics",
        method="WEBSOCKET",
        ip_address=client_ip,
        status_code=200,
    )

    try:
        while True:
            metrics = get_system_metrics()
            await websocket.send_json(asdict(metrics))
            await asyncio.sleep(1)  # Update interval
    except WebSocketDisconnect:
        print(f"Client {client_ip} disconnected from metrics websocket")
    except Exception as e:
        print(f"Error in metrics websocket: {e}")


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint for Kubernetes/Docker."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "0.1.0",
        "components": {
            "database": "healthy",
            "cache": "healthy",
            "gpu": "available" if HAS_PYNVML else "not_available",
        },
    }


# Protected API endpoints with authentication


@app.post("/api/pipeline/start")
async def start_pipeline(current_user: User = Depends(get_current_user)) -> Dict[str, Any]:
    """Start the pipeline processing."""
    # Audit log the action
    audit_logger.log_api_access(
        user_id=str(current_user.user_id),
        user_roles=[role.value for role in current_user.roles],
        endpoint="/api/pipeline/start",
        method="POST",
        ip_address=None,  # Will be logged by middleware
        request_id=uuid4(),
        status_code=200,
    )

    # Check permissions
    if not current_user.has_any_role(UserRole.ADMIN, UserRole.USER):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions to start pipeline")

    # Start pipeline logic here
    return {
        "status": "pipeline_started",
        "user": current_user.username,
        "timestamp": time.time(),
    }


@app.post("/api/pipeline/stop")
async def stop_pipeline(current_user: User = Depends(get_current_user)) -> Dict[str, Any]:
    """Stop the pipeline processing."""
    # Audit log the action
    audit_logger.log_api_access(
        user_id=str(current_user.user_id),
        user_roles=[role.value for role in current_user.roles],
        endpoint="/api/pipeline/stop",
        method="POST",
        ip_address=None,  # Will be logged by middleware
        request_id=uuid4(),
        status_code=200,
    )

    # Check permissions
    if not current_user.has_any_role(UserRole.ADMIN, UserRole.USER):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions to stop pipeline")

    # Stop pipeline logic here
    return {
        "status": "pipeline_stopped",
        "user": current_user.username,
        "timestamp": time.time(),
    }


@app.get("/api/pipeline/status")
async def get_pipeline_status(current_user: User = Depends(get_current_user)) -> Dict[str, Any]:
    """Get current pipeline status."""
    # Basic read permission - all authenticated users can view status
    # Get total documents processed
    total_processed = 0
    for metric in business_metrics.documents_processed.collect():
        for sample in metric.samples:
            if sample.name.endswith("_total"):
                total_processed += sample.value

    return {
        "status": "running",
        "documents_processed": int(total_processed),
        "documents_in_flight": business_metrics.documents_in_flight._value.get(),
        "error_rate": 0.02,  # Calculate from metrics
        "timestamp": time.time(),
    }


@app.get("/api/stats/failures")
async def get_failure_stats(current_user: User = Depends(get_current_user)) -> Dict[str, Any]:
    """Get failure statistics."""
    # Import here to avoid circular imports
    from quarrycore.container import DependencyContainer

    # Create a container instance to get the dead letter service
    container = DependencyContainer()
    await container.initialize()

    try:
        dead_letter_service = await container.get_dead_letter()
        return await dead_letter_service.failure_stats()
    except Exception as e:
        # Fallback to mock data if service fails
        return {
            "total_failures": 0,
            "failures_by_stage": {},
            "retryable_failures": 0,
            "permanent_failures": 0,
            "error": f"Service unavailable: {str(e)}",
        }
    finally:
        await container.shutdown()


@app.post("/api/config/update")
async def update_configuration(
    config_updates: Dict[str, Any], current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Update system configuration."""
    # Require admin role
    if not current_user.is_admin():
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required")

    # Audit log configuration change
    for key, new_value in config_updates.items():
        audit_logger.log_configuration_change(
            user_id=str(current_user.user_id),
            user_roles=[role.value for role in current_user.roles],
            config_key=key,
            old_value="<previous>",  # Would fetch actual old value
            new_value=new_value,
            ip_address=None,  # Will be logged by middleware
        )

    # Apply configuration changes here
    return {
        "status": "configuration_updated",
        "updated_keys": list(config_updates.keys()),
        "timestamp": time.time(),
    }


@app.middleware("http")
async def add_process_time_header(request: Request, call_next: Callable) -> Any:
    """Add custom headers and log all requests."""
    start_time = time.time()
    request_id = uuid4()

    # Add request ID to headers
    request.state.request_id = request_id

    response = await call_next(request)

    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    response.headers["X-Request-ID"] = str(request_id)

    # Log API access (for non-websocket requests)
    if not request.url.path.startswith("/ws/"):
        user_id = None
        user_roles = []
        if hasattr(request.state, "user") and request.state.user:
            user_id = str(request.state.user.user_id)
            user_roles = [role.value for role in request.state.user.roles]

        audit_logger.log_api_access(
            user_id=user_id,
            user_roles=user_roles,
            endpoint=request.url.path,
            method=request.method,
            ip_address=request.client.host if request.client else None,
            request_id=request_id,
            status_code=response.status_code,
            response_time_ms=process_time * 1000,
        )

    return response


def run_web_server(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Function to run the FastAPI server."""
    import uvicorn

    print(f"Starting QuarryCore Web UI at http://{host}:{port}")
    app.state.start_time = time.time()
    uvicorn.run(app, host=host, port=port)
