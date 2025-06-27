#!/usr/bin/env python3
"""
Dummy application for signal handling integration tests.

This module runs a Pipeline with stub URLs and can be interrupted
with signals to test graceful shutdown and checkpoint creation.
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

from quarrycore.container import DependencyContainer
from quarrycore.pipeline import Pipeline, PipelineSettings


async def main(hold_seconds: int = 5):
    """Run a dummy pipeline that can be interrupted."""
    print("Starting dummy pipeline...")

    # Configure checkpoint directory from env or default
    checkpoint_dir = Path(os.environ.get("CHECKPOINT_DIR", "test-checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    settings = PipelineSettings(
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=1.0,  # Fast checkpointing for tests
    )

    # Create container and pipeline
    container = DependencyContainer()
    pipeline = Pipeline(container, settings=settings)

    # Create some test URLs - enough to keep it busy for a while
    test_urls = [
        f"https://example{i % 5}.com/page{i}"
        for i in range(500)  # More URLs to ensure we can interrupt during processing
    ]

    try:
        # Run pipeline - it will handle signals internally
        result = await pipeline.run(urls=test_urls, batch_size=5)  # Smaller batch for more frequent checkpointing
        print(f"Pipeline completed: {result}")

        # Only sleep if pipeline completed without interruption and hold_seconds > 0
        if result.get("status") == "completed" and hold_seconds > 0:
            print("Starting QuarryCore Web Dashboard...")
            # Keep running for the specified time to allow interruption
            await asyncio.sleep(hold_seconds)
    except KeyboardInterrupt:
        print("Shutdown requested")
        raise  # Re-raise to let pipeline handle it properly
    except asyncio.CancelledError:
        print("Shutdown requested")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dummy pipeline for testing")
    parser.add_argument(
        "--hold-seconds", type=int, default=5, help="Seconds to keep running after pipeline completes (default: 5)"
    )
    args = parser.parse_args()

    # Set test mode environment variable
    os.environ["QUARRY_TEST_MODE"] = "1"

    try:
        asyncio.run(main(args.hold_seconds))
    except KeyboardInterrupt:
        # Signal handled, exit gracefully
        sys.exit(0)
