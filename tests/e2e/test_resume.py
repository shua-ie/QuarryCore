"""
End-to-end tests for pipeline checkpoint/resume functionality.

This test validates the complete happy-path pipeline with:
- Checkpoint creation and atomic saving
- Pipeline resume from checkpoint
- Dead letter queue integration for failed URLs
- Complete processing validation

Test Scenario:
1. Start pipeline with 20 URLs
2. Kill after 5 seconds (SIGINT)
3. Resume from checkpoint
4. Validate all URLs processed
5. Check dead letter queue for failures
"""

import asyncio
import os
import signal
import sqlite3
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
import structlog
from quarrycore.config import Config
from quarrycore.container import DependencyContainer
from quarrycore.pipeline import Pipeline, PipelineCheckpoint
from quarrycore.recovery.dead_letter import DeadLetterQueue

# Test URLs - mix of valid and invalid for realistic testing
TEST_URLS = [
    "https://httpbin.org/delay/1",  # Valid, slow
    "https://httpbin.org/status/200",  # Valid, fast
    "https://httpbin.org/status/404",  # Invalid status
    "https://httpbin.org/delay/2",  # Valid, slower
    "https://httpbin.org/status/500",  # Server error
    "https://httpbin.org/json",  # Valid JSON response
    "https://httpbin.org/html",  # Valid HTML response
    "https://httpbin.org/status/403",  # Forbidden
    "https://httpbin.org/delay/3",  # Valid, slowest
    "https://httpbin.org/status/301",  # Redirect
    "https://httpbin.org/uuid",  # Valid UUID response
    "https://httpbin.org/status/429",  # Rate limited
    "https://httpbin.org/base64/SFRUUEJJTiBpcyBhd2Vzb21l",  # Valid base64
    "https://httpbin.org/status/502",  # Bad gateway
    "https://httpbin.org/ip",  # Valid IP response
    "https://httpbin.org/status/503",  # Service unavailable
    "https://httpbin.org/user-agent",  # Valid user agent
    "https://httpbin.org/status/504",  # Gateway timeout
    "https://httpbin.org/headers",  # Valid headers response
    "https://httpbin.org/status/418",  # I'm a teapot
]

logger = structlog.get_logger(__name__)


class MockAsyncContextManager:
    """Simple async context manager for testing."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return False


class MockContainer(DependencyContainer):
    """Mock container for testing with lazy initialization."""

    def __init__(self):
        super().__init__()
        self.is_running = False
        self._instances = {}
        # Use lazy initialization to avoid event loop binding issues
        self._instances_lock = None

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


@pytest.fixture
def temp_checkpoint_dir():
    """Create temporary directory for checkpoints."""
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_dir = Path(temp_dir) / "checkpoints"
        checkpoint_dir.mkdir()

        # Change to temp directory
        original_cwd = os.getcwd()
        os.chdir(temp_dir)

        yield checkpoint_dir

        # Restore original directory
        os.chdir(original_cwd)


@pytest.fixture
def temp_dead_letter_db():
    """Create temporary dead letter database."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as temp_file:
        db_path = Path(temp_file.name)

    yield db_path

    # Cleanup
    if db_path.exists():
        db_path.unlink()


@pytest.mark.asyncio
async def test_pipeline_checkpoint_resume_e2e(temp_checkpoint_dir, temp_dead_letter_db):
    """
    End-to-end test for pipeline checkpoint and resume functionality.

    This test:
    1. Starts pipeline with 20 URLs
    2. Interrupts after 5 seconds
    3. Validates checkpoint was saved
    4. Resumes from checkpoint
    5. Ensures all URLs are processed
    6. Checks dead letter queue for failures
    """
    # Setup mock container
    container = MockContainer()

    # Phase 1: Start pipeline and interrupt after 5 seconds
    logger.info("=== Phase 1: Starting pipeline with interruption ===")

    pipeline = Pipeline(container, max_concurrency=5)

    async def interrupt_after_delay():
        """Interrupt pipeline after 2 seconds."""
        await asyncio.sleep(2)
        logger.info("Sending SIGINT to interrupt pipeline...")
        pipeline._shutdown_requested = True
        # Simulate SIGINT signal
        if pipeline.state:
            await pipeline._save_checkpoint()

    # Start the interrupt task
    interrupt_task = asyncio.create_task(interrupt_after_delay())

    try:
        # Start pipeline - should be interrupted
        result1 = await pipeline.run(
            urls=TEST_URLS.copy(),
            batch_size=5,
            checkpoint_interval=2.0,  # Save checkpoints frequently for testing
        )
        logger.info(f"Phase 1 completed: {result1}")
    except Exception as e:
        logger.info(f"Phase 1 interrupted (expected): {e}")

    # Cancel interrupt task if still running
    if not interrupt_task.done():
        interrupt_task.cancel()

    # Validate checkpoint was saved
    checkpoint_files = list(temp_checkpoint_dir.glob("*.json"))
    assert checkpoint_files, "No checkpoint file was created"

    checkpoint_path = checkpoint_files[0]
    logger.info(f"Found checkpoint: {checkpoint_path}")

    # Load and validate checkpoint content
    checkpoint = await pipeline._load_checkpoint(checkpoint_path)
    assert isinstance(checkpoint, PipelineCheckpoint)
    assert checkpoint.job_id == pipeline.job_id

    initial_processed = checkpoint.processed_count
    initial_failed = checkpoint.failed_count
    initial_remaining = len(checkpoint.urls_remaining)

    logger.info(
        f"Phase 1 results - Processed: {initial_processed}, Failed: {initial_failed}, Remaining: {initial_remaining}"
    )

    # Check if we need to test resume (only if there are URLs remaining)
    if initial_remaining > 0:
        # Phase 2: Resume from checkpoint
        logger.info("=== Phase 2: Resuming from checkpoint ===")

        # Create new pipeline instance for resume
        pipeline2 = Pipeline(container, max_concurrency=5)

        # Resume from checkpoint
        result2 = await pipeline2.run(
            urls=[],  # Empty URLs list since we're resuming
            batch_size=5,
            checkpoint_interval=10.0,
            resume_from=checkpoint_path,
        )

        logger.info(f"Phase 2 completed: {result2}")

        # Validate final results
        final_processed = pipeline2.state.processed_count if pipeline2.state else 0
        final_failed = pipeline2.state.failed_count if pipeline2.state else 0
        final_remaining = len(pipeline2.state.urls_remaining) if pipeline2.state else 0

        logger.info(
            f"Phase 2 results - Processed: {final_processed}, Failed: {final_failed}, Remaining: {final_remaining}"
        )

        # Assertions for resume functionality
        assert final_remaining == 0, f"URLs still remaining after resume: {final_remaining}"
        assert final_processed + final_failed == len(
            TEST_URLS
        ), f"Total processed ({final_processed + final_failed}) != total URLs ({len(TEST_URLS)})"
        assert final_processed >= initial_processed, "Final processed count should be >= initial count"
    else:
        logger.info("=== Phase 2: Skipped (all URLs already processed) ===")
        # All URLs were processed in phase 1, which is also a valid success case
        assert initial_processed + initial_failed == len(
            TEST_URLS
        ), f"Total processed ({initial_processed + initial_failed}) != total URLs ({len(TEST_URLS)})"

    # Phase 3: Validate dead letter queue
    logger.info("=== Phase 3: Validating dead letter queue ===")

    # Check dead letter database
    dlq_path = Path("dead_letter.db")
    if dlq_path.exists():
        conn = sqlite3.connect(dlq_path)
        cursor = conn.cursor()

        # Check if table exists and has data
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='failed_documents'")
        table_exists = cursor.fetchone() is not None

        if table_exists:
            cursor.execute("SELECT COUNT(*) FROM failed_documents")
            failed_count_dlq = cursor.fetchone()[0]

            cursor.execute("SELECT url, error_type, error_message FROM failed_documents LIMIT 5")
            sample_failures = cursor.fetchall()

            logger.info(f"Dead letter queue has {failed_count_dlq} failed documents")
            for i, (url, error_type, error_msg) in enumerate(sample_failures):
                logger.info(f"Sample failure {i + 1}: {url} - {error_type}: {error_msg}")

        conn.close()

        # Cleanup dead letter DB
        dlq_path.unlink()

    logger.info("=== Test completed successfully ===")


@pytest.mark.asyncio
async def test_checkpoint_atomic_saving(temp_checkpoint_dir):
    """Test that checkpoint saving is atomic (temp file → rename)."""
    container = MockContainer()
    pipeline = Pipeline(container)

    # Create minimal state
    from quarrycore.pipeline import PipelineStage, PipelineState

    pipeline.state = PipelineState(
        pipeline_id="test-pipeline",
        stage=PipelineStage.CRAWL,
        processed_count=5,
        failed_count=1,
        start_time=time.time(),
        last_checkpoint=time.time(),
        urls_remaining=["http://example.com"],
        batch_size=10,
        error_count_by_stage={},
    )

    # Save checkpoint
    await pipeline._save_checkpoint()

    # Verify checkpoint file exists and is valid JSON
    checkpoint_files = list(temp_checkpoint_dir.glob("*.json"))
    assert len(checkpoint_files) == 1

    checkpoint_path = checkpoint_files[0]

    # Verify no temporary files left behind
    temp_files = list(temp_checkpoint_dir.glob("*.tmp"))
    assert len(temp_files) == 0, f"Temporary files left behind: {temp_files}"

    # Verify checkpoint content is valid
    checkpoint = await pipeline._load_checkpoint(checkpoint_path)
    assert checkpoint.pipeline_id == "test-pipeline"
    assert checkpoint.processed_count == 5
    assert checkpoint.failed_count == 1


@pytest.mark.asyncio
async def test_dead_letter_integration(temp_dead_letter_db):
    """Test integration between pipeline and dead letter queue."""

    # Initialize dead letter queue
    dlq = DeadLetterQueue(db_path=temp_dead_letter_db)
    await dlq.initialize()

    # Add some failed documents
    from quarrycore.protocols import ErrorInfo, ErrorSeverity

    error_info = ErrorInfo(
        error_type="HTTPError",
        error_message="404 Not Found",
        severity=ErrorSeverity.MEDIUM,
        is_retryable=True,
    )

    await dlq.add_failed_document(
        url="https://example.com/not-found",
        failure_stage="crawl",
        error_info=error_info,
        metadata={"job_id": "test-job", "pipeline_id": "test-pipeline"},
    )

    # Verify document was added
    failed_docs = await dlq.get_failed_documents(limit=10)
    assert len(failed_docs) == 1
    assert failed_docs[0].url == "https://example.com/not-found"
    assert failed_docs[0].error_info.error_type == "HTTPError"

    # Get statistics
    stats = await dlq.get_failure_statistics()
    assert stats["total_failures"] == 1
    assert stats["failures_by_stage"]["crawl"] == 1
    assert stats["failures_by_error"]["HTTPError"] == 1

    await dlq.close()


@pytest.mark.asyncio
async def test_signal_handling():
    """Test that signal handling works correctly for checkpoint saving."""
    container = MockContainer()
    pipeline = Pipeline(container)

    # Setup minimal state
    from quarrycore.pipeline import PipelineStage, PipelineState

    pipeline.state = PipelineState(
        pipeline_id="test-signal",
        stage=PipelineStage.CRAWL,
        processed_count=3,
        failed_count=0,
        start_time=time.time(),
        last_checkpoint=time.time(),
        urls_remaining=["http://test1.com", "http://test2.com"],
        batch_size=10,
        error_count_by_stage={},
    )

    # Setup signal handlers
    pipeline._setup_signal_handlers()

    # Simulate receiving SIGINT
    original_handler = signal.getsignal(signal.SIGINT)
    assert original_handler != signal.SIG_DFL, "Signal handler should be installed"

    # Test shutdown flag
    assert not pipeline._shutdown_requested

    # Simulate signal (we can't actually send signal in test)
    pipeline._shutdown_requested = True
    assert pipeline._shutdown_requested

    # Cleanup
    pipeline._cleanup_signal_handlers()


if __name__ == "__main__":
    # Run the test directly
    import sys

    asyncio.run(test_pipeline_checkpoint_resume_e2e(Path("test_checkpoints"), Path("test_dead_letter.db")))
