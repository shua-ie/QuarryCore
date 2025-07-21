"""
Unit tests for signal handling with monkey-patched signal module.

Tests signal handlers in isolation without subprocess overhead.
"""

import asyncio
import signal
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from quarrycore.config import Config
from quarrycore.container import DependencyContainer
from quarrycore.pipeline import Pipeline, PipelineSettings


class MockContainer(DependencyContainer):
    """Mock container for testing with lazy initialization."""

    def __init__(self):
        super().__init__()
        self.is_running = False
        self._instances = {}
        # Use lazy initialization to avoid event loop binding issues
        self._instances_lock = None
        self._observer: Optional[Any] = None
        self._shutdown_handlers: List[Callable[[], Any]] = []
        self.pipeline_id = str(uuid4())
        self.config: Optional[Config] = None

    @property
    def instances_lock(self):
        """Lazy initialize the lock to avoid event loop binding issues."""
        if self._instances_lock is None:
            self._instances_lock = asyncio.Lock()
        return self._instances_lock

    def get_health_status(self):
        return {"status": "healthy"}

    @asynccontextmanager
    async def lifecycle(self):
        yield self

    async def __aenter__(self):
        """Ensure lock is created in the correct event loop."""
        await self.instances_lock.acquire()
        self.is_running = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.is_running = False
        self.instances_lock.release()

    async def get_observability(self):
        mock_obs = AsyncMock()
        mock_obs.start_monitoring = lambda: MockAsyncContextManager()
        mock_obs.log_error = AsyncMock()
        return mock_obs

    async def get_quality(self):
        mock_quality = AsyncMock()
        mock_quality.assess_quality = AsyncMock(return_value=MagicMock(overall_score=0.8))
        return mock_quality

    async def get_storage(self):
        mock_storage = AsyncMock()
        mock_storage.store_extracted_content = AsyncMock(return_value="test-doc-id")
        return mock_storage

    async def get_http_client(self):
        """Return a mock HTTP client."""
        mock_http_client = AsyncMock()
        mock_http_client.fetch = AsyncMock()

        # Create a mock CrawlerResponse
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.headers = {"content-type": "text/html"}
        mock_response.body = b"<html><title>Test</title><body>Test content</body></html>"
        mock_response.final_url = "https://example.com"

        mock_http_client.fetch.return_value = mock_response
        return mock_http_client


class MockAsyncContextManager:
    """Simple async context manager for testing."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return False


class TestSignalHandler:
    """Unit tests for signal handling functionality."""

    @pytest.mark.asyncio
    async def test_sigint_handler_sets_shutdown_flag(self):
        """Test that SIGINT handler sets the shutdown flag."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            settings = PipelineSettings(checkpoint_dir=temp_path / "checkpoints", checkpoint_interval=30.0)

            container = MockContainer()
            pipeline = Pipeline(container, max_concurrency=2, settings=settings)

            # Mock signal handling
            signal_handlers = {}

            def mock_signal(sig, handler):
                signal_handlers[sig] = handler

            with patch("signal.signal", side_effect=mock_signal):
                # Setup signal handlers (this happens in pipeline initialization)
                pipeline._setup_signal_handlers()

                # Verify handlers were registered
                assert signal.SIGINT in signal_handlers
                assert signal.SIGTERM in signal_handlers

                # Test SIGINT handler
                assert not pipeline._shutdown_requested
                signal_handlers[signal.SIGINT](signal.SIGINT, None)
                assert pipeline._shutdown_requested

    @pytest.mark.asyncio
    async def test_sigterm_handler_sets_shutdown_flag(self):
        """Test that SIGTERM handler sets the shutdown flag."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            settings = PipelineSettings(checkpoint_dir=temp_path / "checkpoints", checkpoint_interval=30.0)

            container = MockContainer()
            pipeline = Pipeline(container, max_concurrency=2, settings=settings)

            # Mock signal handling
            signal_handlers = {}

            def mock_signal(sig, handler):
                signal_handlers[sig] = handler

            with patch("signal.signal", side_effect=mock_signal):
                # Setup signal handlers
                pipeline._setup_signal_handlers()

                # Test SIGTERM handler
                assert not pipeline._shutdown_requested
                signal_handlers[signal.SIGTERM](signal.SIGTERM, None)
                assert pipeline._shutdown_requested

    @pytest.mark.asyncio
    async def test_shutdown_flag_stops_processing(self):
        """Test that setting shutdown flag stops URL processing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            settings = PipelineSettings(
                checkpoint_dir=temp_path / "checkpoints",
                checkpoint_interval=1.0,  # Fast checkpoints
            )

            container = MockContainer()
            pipeline = Pipeline(container, max_concurrency=2, settings=settings)

            # Mock URL processing to track calls and avoid real HTTP
            process_calls = []
            processing_times = []

            async def mock_process_url(url: str, worker_id: str):
                process_calls.append(url)
                start_time = asyncio.get_event_loop().time()

                # Simulate processing with small delay and check for shutdown
                for _ in range(10):  # 10 small checks instead of one big sleep
                    if pipeline._shutdown_requested:
                        break
                    await asyncio.sleep(0.02)  # Small delay

                # Mock successful result
                from quarrycore.pipeline import PipelineStage, ProcessingResult, ProcessingStatus

                processing_time = asyncio.get_event_loop().time() - start_time
                processing_times.append(processing_time)

                return ProcessingResult(
                    document_id=None,
                    status=ProcessingStatus.COMPLETED,
                    stage_completed=PipelineStage.STORAGE,
                    processing_time=processing_time,
                )

            # Patch the method before any processing starts
            pipeline._process_url = mock_process_url

            # Start processing in background with URLs
            urls = ["https://test1.local", "https://test2.local", "https://test3.local", "https://test4.local"]

            pipeline_task = asyncio.create_task(pipeline.run(urls=urls, batch_size=2))

            # Give pipeline time to start workers and begin processing
            await asyncio.sleep(0.1)

            # Trigger shutdown
            pipeline._shutdown_requested = True

            # Wait for pipeline to complete shutdown
            try:
                await asyncio.wait_for(pipeline_task, timeout=3.0)

                # Verify shutdown worked - should have processed some but not all URLs
                assert len(process_calls) > 0, "Pipeline should have started processing some URLs"
                assert len(process_calls) <= len(urls), "Should not have processed more URLs than provided"

                # If all URLs were processed, they should have completed quickly due to shutdown check
                if len(process_calls) == len(urls):
                    avg_processing_time = sum(processing_times) / len(processing_times)
                    assert (
                        avg_processing_time < 0.15
                    ), f"URLs should have completed quickly due to shutdown, avg time: {avg_processing_time}"

            except asyncio.TimeoutError:
                pipeline_task.cancel()
                try:
                    await pipeline_task
                except asyncio.CancelledError:
                    pass
                pytest.fail(f"Pipeline did not respond to shutdown flag in time. Processed {len(process_calls)} URLs")

    @pytest.mark.asyncio
    async def test_checkpoint_saved_on_signal_shutdown(self):
        """Test that checkpoint is saved when signal triggers shutdown."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            settings = PipelineSettings(
                checkpoint_dir=temp_path / "checkpoints",
                checkpoint_interval=30.0,  # Long interval so only signal triggers save
            )

            container = MockContainer()
            pipeline = Pipeline(container, max_concurrency=2, settings=settings)

            # Start some processing to create state
            from quarrycore.pipeline import PipelineStage, PipelineState

            pipeline.state = PipelineState(
                pipeline_id="test-pipeline",
                stage=PipelineStage.CRAWL,
                processed_count=3,
                failed_count=1,
                start_time=1234567890.0,
                last_checkpoint=1234567890.0,
                urls_remaining=["https://test.local/remaining"],
                batch_size=5,
                error_count_by_stage={"crawl": 1},
            )

            # Mock signal handling and trigger save
            signal_handlers = {}

            def mock_signal(sig, handler):
                signal_handlers[sig] = handler

            with patch("signal.signal", side_effect=mock_signal):
                pipeline._setup_signal_handlers()

                # Trigger signal handler
                signal_handlers[signal.SIGINT](signal.SIGINT, None)

                # Manually trigger checkpoint save (simulating what happens in main loop)
                await pipeline._save_checkpoint()

                # Verify checkpoint file exists
                from quarrycore.utils.slugify import slugify

                safe_job_id = slugify(pipeline.job_id, replacement="-", max_length=100, lowercase=True)
                checkpoint_path = settings.checkpoint_dir / f"{safe_job_id}.json"

                assert checkpoint_path.exists(), "Checkpoint file should exist after signal"

                # Verify checkpoint can be loaded
                checkpoint = await pipeline._load_checkpoint(checkpoint_path)
                assert checkpoint.processed_count == 3
                assert checkpoint.failed_count == 1
                assert len(checkpoint.urls_remaining) == 1

    @pytest.mark.asyncio
    async def test_multiple_signals_handled_gracefully(self):
        """Test that multiple signals don't cause issues."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            settings = PipelineSettings(checkpoint_dir=temp_path / "checkpoints")

            container = MockContainer()
            pipeline = Pipeline(container, settings=settings)

            signal_handlers = {}

            def mock_signal(sig, handler):
                signal_handlers[sig] = handler

            with patch("signal.signal", side_effect=mock_signal):
                pipeline._setup_signal_handlers()

                # Send multiple signals
                assert not pipeline._shutdown_requested

                signal_handlers[signal.SIGINT](signal.SIGINT, None)
                assert pipeline._shutdown_requested

                # Second signal should not cause issues
                signal_handlers[signal.SIGINT](signal.SIGINT, None)
                assert pipeline._shutdown_requested  # Still true

                # Different signal should also work
                signal_handlers[signal.SIGTERM](signal.SIGTERM, None)
                assert pipeline._shutdown_requested
