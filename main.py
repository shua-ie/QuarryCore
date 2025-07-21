#!/usr/bin/env python3
"""
Production entry point for QuarryCore.

This file serves as the main entry point for production deployments,
providing health checks, graceful shutdown, and proper error handling.
"""

from __future__ import annotations

import asyncio
import json

# Configure basic logging for production
import logging
import os
import signal
import sys
from pathlib import Path

import structlog
from quarrycore.container import DependencyContainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

logger = structlog.get_logger(__name__)

# Global state for graceful shutdown
running = True
current_container: DependencyContainer | None = None


def signal_handler(signum: int, frame) -> None:
    """Handle shutdown signals gracefully."""
    global running, current_container
    logger.info(f"Received signal {signum}, initiating graceful shutdown")
    running = False

    if current_container:
        asyncio.create_task(current_container.shutdown())


# Register signal handlers
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)


async def health_check() -> dict:
    """Perform health check for container orchestration."""
    try:
        container = DependencyContainer()
        return {
            "status": "healthy",
            "timestamp": asyncio.get_event_loop().time(),
            **container.get_health_status(),
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": asyncio.get_event_loop().time(),
        }


async def run_production_pipeline():
    """Run the pipeline in production mode."""
    global current_container, running

    # Load configuration from environment or default location
    config_path = os.getenv("QUARRYCORE_CONFIG")
    config_path_obj = Path(config_path) if config_path else None

    current_container = DependencyContainer(config_path_obj)

    try:
        async with current_container.lifecycle():
            logger.info("QuarryCore production pipeline starting")

            # Start monitoring
            observability = await current_container.get_observability()

            async with observability.start_monitoring():
                # Production pipeline would read from a queue, database, or API
                # For now, we'll simulate with a health check loop
                while running:
                    health = await health_check()
                    logger.info("Health check", **health)

                    # In production, this would process actual work
                    await asyncio.sleep(30)

    except Exception as e:
        logger.error("Production pipeline failed", error=str(e))
        raise
    finally:
        logger.info("QuarryCore production pipeline stopped")


async def main():
    """Main entry point."""
    # Check if this is a health check request
    if len(sys.argv) > 1 and sys.argv[1] == "health":
        health = await health_check()
        print(json.dumps(health, indent=2))
        sys.exit(0 if health["status"] == "healthy" else 1)

    # Run the production pipeline
    try:
        await run_production_pipeline()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down")
    except Exception as e:
        logger.error("Unhandled exception in main", error=str(e))
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
