#!/usr/bin/env python3
"""
Dummy application for signal handling integration tests.

This module runs a Pipeline with stub URLs and can be interrupted
with signals to test graceful shutdown and checkpoint creation.
"""

import argparse
import asyncio
import os
import signal
import sys
from pathlib import Path
from typing import Optional

# We'll let the Pipeline handle signals, not install our own conflicting handlers


async def main(hold_seconds: int = 5):
    """Run a dummy pipeline that can be interrupted."""
    print("Starting dummy pipeline...", flush=True)

    # Configure checkpoint directory from env or default
    checkpoint_dir = Path(os.environ.get("CHECKPOINT_DIR", "test-checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Import pipeline components after basic setup
    from quarrycore.container import DependencyContainer
    from quarrycore.pipeline import Pipeline, PipelineSettings

    settings = PipelineSettings(
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=0.5,  # Very fast checkpointing for tests
    )

    # Create container and pipeline
    container = DependencyContainer()
    pipeline = Pipeline(container, settings=settings)

    # Create some test URLs - enough to keep it busy for a while
    test_urls = [
        f"https://example{i % 5}.com/page{i}"
        for i in range(50)  # Further reduced to ensure we're still processing when signal arrives
    ]

    # Create a shutdown event that can be set by signal handlers
    shutdown_event = asyncio.Event()

    # Install async-compatible signal handlers
    loop = asyncio.get_event_loop()

    def signal_handler(signum):
        print(f"Received signal {signum} in dummy app signal handler", flush=True)
        shutdown_event.set()

    # Use asyncio's add_signal_handler for proper async signal handling
    loop.add_signal_handler(signal.SIGINT, lambda: signal_handler(signal.SIGINT))
    loop.add_signal_handler(signal.SIGTERM, lambda: signal_handler(signal.SIGTERM))

    print("Signal handlers installed", flush=True)

    try:
        # Signal that we're ready to handle interrupts
        print("READY: Signal handlers installed, pipeline starting...", flush=True)

        # Add a small delay to ensure the test can catch the READY signal
        await asyncio.sleep(0.1)

        # Run pipeline with concurrent shutdown monitoring
        pipeline_task = asyncio.create_task(
            pipeline.run(urls=test_urls, batch_size=1)  # Back to batch size 1 for slower processing
        )
        shutdown_task = asyncio.create_task(shutdown_event.wait())

        # Wait for either pipeline completion or shutdown signal
        done, pending = await asyncio.wait({pipeline_task, shutdown_task}, return_when=asyncio.FIRST_COMPLETED)

        if shutdown_task in done:
            # Shutdown was requested
            print("Shutdown requested, signaling pipeline...", flush=True)

            # Set the pipeline's shutdown flag directly
            pipeline._shutdown_requested = True
            if pipeline._shutdown_event:
                pipeline._shutdown_event.set()

            # Give pipeline a moment to save checkpoint
            await asyncio.sleep(0.5)

            # Cancel the pipeline task
            pipeline_task.cancel()
            try:
                await asyncio.wait_for(pipeline_task, timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            print("Pipeline cancelled, exiting gracefully", flush=True)

            # Ensure checkpoint was saved
            if pipeline.state:
                await pipeline._save_checkpoint()
                print("Final checkpoint saved", flush=True)
            return

        # Pipeline completed normally
        result = await pipeline_task
        print(f"Pipeline completed: {result}", flush=True)

        # Only sleep if pipeline completed without interruption and hold_seconds > 0
        if result.get("status") == "completed" and hold_seconds > 0 and not shutdown_event.is_set():
            print("Starting QuarryCore Web Dashboard...", flush=True)
            # Keep running for the specified time to allow interruption
            try:
                await asyncio.wait_for(shutdown_event.wait(), timeout=hold_seconds)
                print("Shutdown requested during hold period", flush=True)
            except asyncio.TimeoutError:
                print("Hold period completed, exiting normally", flush=True)

    except Exception as e:
        print(f"Error in dummy app: {e}", flush=True)
        raise
    finally:
        print("Dummy app exiting", flush=True)
        # Remove signal handlers
        loop.remove_signal_handler(signal.SIGINT)
        loop.remove_signal_handler(signal.SIGTERM)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dummy pipeline for testing")
    parser.add_argument(
        "--hold-seconds", type=int, default=5, help="Seconds to keep running after pipeline completes (default: 5)"
    )
    args = parser.parse_args()

    # Set test mode environment variable
    os.environ["QUARRY_TEST_MODE"] = "1"

    # Configure observability for testing to avoid port conflicts
    os.environ["QUARRY_MONITORING__PROMETHEUS_PORT"] = "0"  # Use dynamic port
    os.environ["QUARRY_MONITORING__WEB_UI__ENABLED"] = "false"  # Disable web UI
    os.environ["QUARRY_MONITORING__ENABLED"] = "false"  # Disable observability entirely for tests

    # Configure HTTP client for slower processing to allow signal interruption
    os.environ["QUARRY_CRAWLER__HTTP_CLIENT__TIMEOUT"] = "2.0"  # Longer timeout per request
    os.environ["QUARRY_CRAWLER__HTTP_CLIENT__MAX_RETRIES"] = "5"  # More retries
    os.environ["QUARRY_CRAWLER__HTTP_CLIENT__RETRY_DELAY"] = "1.0"  # Longer delay between retries
    os.environ["QUARRY_PIPELINE__WORKER_DELAY"] = "0.1"  # Add delay between URL processing

    # Run with debug mode for better KeyboardInterrupt propagation
    try:
        asyncio.run(main(args.hold_seconds))
        sys.exit(0)  # Ensure clean exit
    except KeyboardInterrupt:
        print("KeyboardInterrupt caught at top level", flush=True)
        sys.exit(0)
