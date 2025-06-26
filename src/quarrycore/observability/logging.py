"""
Configures structured logging for the application using structlog.
"""
from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING, Dict, Any, List

import structlog
from structlog.types import Processor

if TYPE_CHECKING:
    from quarrycore.config.config import MonitoringConfig

# --- Custom Processors ---

# This is a workaround for the linter, which doesn't know about the `structlog` package.
# The code is correct and will work at runtime.
if TYPE_CHECKING:
    from structlog.types import Processor
else:
    Processor = object

def add_correlation_id(logger: logging.Logger, method_name: str, event_dict: Dict[Any, Any]) -> Dict[Any, Any]:
    """
    Adds a correlation_id to the log record if it's in the context.
    This would be set by a middleware or context manager in a real app.
    """
    from structlog.contextvars import get_contextvars
    ctx = get_contextvars()
    if "correlation_id" in ctx:
        event_dict["correlation_id"] = ctx["correlation_id"]
    return event_dict
    
# --- Configuration ---

def configure_logging(config: MonitoringConfig) -> None:
    """
    Sets up structlog to handle all logging for the application.
    """
    shared_processors: List[Any] = [  # Using Any for Processor compatibility
        structlog.contextvars.merge_contextvars,
        add_correlation_id,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
    ]

    log_renderer: Any  # Using Any for Processor compatibility
    if config.log_file:
        # Structured JSON logging for production/file output
        log_renderer = structlog.processors.JSONRenderer()
        handler: logging.Handler = logging.FileHandler(config.log_file)
    else:
        # More readable console output for development
        log_renderer = structlog.dev.ConsoleRenderer(colors=True)
        handler = logging.StreamHandler(sys.stdout)

    # Configure the standard logging library to pass records to structlog
    logging.basicConfig(
        format="%(message)s",
        level=config.log_level.upper(),
        handlers=[handler],
    )

    # Configure structlog itself
    structlog.configure(
        processors=shared_processors + [
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # After configuring, get a logger to confirm
    logger = structlog.get_logger("quarrycore.logging")
    logger.info("Logging configured", level=config.log_level, output=config.log_file or "console") 