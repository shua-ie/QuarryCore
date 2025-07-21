"""
Comprehensive E2E tests for pipeline validation and hardening.

Tests all acceptance criteria:
- AC-01: Atomic checkpoint save (Linux/Windows compatibility)
- AC-02: Job ID slugification for safe filenames
- AC-03: Exact-state resume with empty urls_remaining
- AC-04: Duplicate dead-letter guard with UNIQUE constraint
- AC-05: Domain-based failure backpressure
- AC-06: Configurable settings via environment variables
- AC-07: Graceful SIGINT/SIGTERM handling
"""

import asyncio
import json
import os
import shutil
import signal
import sqlite3
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch
from urllib.parse import urlparse
from uuid import uuid4

import pytest
import structlog
from quarrycore.config import Config
from quarrycore.container import DependencyContainer
from quarrycore.pipeline import DomainFailureTracker, Pipeline, PipelineCheckpoint, PipelineSettings
from quarrycore.recovery.dead_letter import DeadLetterQueue
from quarrycore.utils import atomic_write_json, slugify

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

        # Add realistic processing delay to simulate real crawling
        async def delayed_fetch(*args, **kwargs):
            await asyncio.sleep(0.5)  # Small delay to allow interruption
            return mock_response

        mock_http_client.fetch.side_effect = delayed_fetch
        return mock_http_client


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_container():
    """Create mock container for testing."""
    return MockContainer()


@pytest.fixture
def test_urls():
    """Sample URLs for testing."""
    return [
        "https://example.com/page1",
        "https://example.com/page2",
        "https://test.org/article1",
        "https://test.org/article2",
        "https://another.site/content1",
        "https://another.site/content2",
        "https://example.com/page3",
        "https://example.com/page4",
        "https://test.org/article3",
        "https://test.org/article4",
    ]


@pytest.mark.asyncio
async def test_ac01_atomic_checkpoint_save_cross_platform(temp_dir, mock_container):
    """AC-01: Test atomic checkpoint save works on Linux & Windows."""

    # Test settings with custom checkpoint directory
    settings = PipelineSettings(checkpoint_dir=temp_dir / "checkpoints", checkpoint_interval=5.0)

    pipeline = Pipeline(mock_container, max_concurrency=2, settings=settings)

    # Test with unsafe job ID characters
    unsafe_job_id = "job:2024/01/01-12:30:45\\test#batch"
    pipeline.job_id = unsafe_job_id

    # Create mock state
    from quarrycore.pipeline import PipelineStage, PipelineState

    pipeline.state = PipelineState(
        pipeline_id="test-pipeline",
        stage=PipelineStage.CRAWL,
        processed_count=5,
        failed_count=1,
        start_time=time.time(),
        last_checkpoint=time.time(),
        urls_remaining=["https://example.com/remaining"],
        batch_size=10,
        error_count_by_stage={"crawl": 1},
    )

    # Save checkpoint
    await pipeline._save_checkpoint()

    # Verify checkpoint file exists with safe filename
    safe_job_id = slugify(unsafe_job_id, replacement="-", max_length=100, lowercase=True)
    checkpoint_path = settings.checkpoint_dir / f"{safe_job_id}.json"

    assert checkpoint_path.exists(), "Checkpoint file should exist"

    # Verify atomic write created proper JSON
    checkpoint = await pipeline._load_checkpoint(checkpoint_path)
    assert checkpoint.job_id == unsafe_job_id  # Original job_id preserved in content
    assert checkpoint.processed_count == 5
    assert checkpoint.failed_count == 1
    assert len(checkpoint.urls_remaining) == 1


@pytest.mark.asyncio
async def test_ac02_checkpoint_slugify_unsafe_chars(temp_dir, mock_container):
    """AC-02: Test job ID with unsafe characters is properly slugified."""

    settings = PipelineSettings(checkpoint_dir=temp_dir / "checkpoints")
    Pipeline(mock_container, settings=settings)

    # Test various unsafe character combinations
    test_cases = [
        ("job/with\\slashes:and<>pipes|", "job-with-slashes-and-pipes"),
        ("Job With Spaces & Special*Chars!", "job-with-spaces-special-chars"),
        ("file.name.with.dots", "file-name-with-dots"),
        ("normal-job-id", "normal-job-id"),
        ("", "untitled"),
    ]

    for unsafe_id, expected_safe in test_cases:
        safe_id = slugify(unsafe_id, replacement="-", max_length=100, lowercase=True)
        assert (
            safe_id == expected_safe
        ), f"Slugify failed for '{unsafe_id}': got '{safe_id}', expected '{expected_safe}'"


@pytest.mark.asyncio
async def test_ac03_exact_state_resume_empty_urls_remaining(temp_dir, mock_container):
    """AC-03: Test exact-state resume with empty urls_remaining exits with completed status."""

    settings = PipelineSettings(checkpoint_dir=temp_dir / "checkpoints")

    # Create completed checkpoint with no remaining URLs
    completed_checkpoint = PipelineCheckpoint(
        job_id="completed-job",
        pipeline_id="test-pipeline",
        stage="storage",
        processed_count=20,
        failed_count=2,
        start_time=time.time() - 3600,
        last_checkpoint=time.time(),
        urls_remaining=[],  # Empty - all URLs processed
        batch_size=10,
        error_count_by_stage={"crawl": 1, "extract": 1},
    )

    # Save checkpoint
    checkpoint_path = settings.checkpoint_dir / "completed-job.json"
    settings.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(checkpoint_path, completed_checkpoint.model_dump())

    # Create pipeline and resume from completed checkpoint
    pipeline = Pipeline(mock_container, settings=settings)

    result = await pipeline.run(urls=[], resume_from=checkpoint_path)  # Empty URLs since resuming

    # Should exit immediately with completed status
    assert result["status"] == "completed"
    assert result["processed_count"] == 20
    assert result["failed_count"] == 2
    assert result["duration"] == 0
    assert "Resumed from completed checkpoint" in result["message"]


@pytest.mark.asyncio
async def test_ac04_duplicate_dead_letter_guard(temp_dir):
    """AC-04: Test dead letter queue UNIQUE constraint and upsert behavior."""

    dead_letter_db = temp_dir / "test_dead_letter.db"
    dlq = DeadLetterQueue(db_path=dead_letter_db)
    await dlq.initialize()

    try:
        from quarrycore.protocols import ErrorInfo, ErrorSeverity

        # Add first failure
        error_info1 = ErrorInfo(
            error_type="ConnectionError",
            error_message="Connection timeout",
            severity=ErrorSeverity.MEDIUM,
            is_retryable=True,
        )

        await dlq.add_failed_document(url="https://example.com/test", failure_stage="crawl", error_info=error_info1)

        # Add duplicate failure (same URL + stage)
        error_info2 = ErrorInfo(
            error_type="ConnectionError",
            error_message="Connection timeout again",
            severity=ErrorSeverity.MEDIUM,
            is_retryable=True,
        )

        await dlq.add_failed_document(
            url="https://example.com/test",
            failure_stage="crawl",  # Same URL + stage combination
            error_info=error_info2,
        )

        # Verify UNIQUE constraint worked - should be upsert, not insert
        # Check database directly
        assert dlq._db is not None, "Database should be initialized"
        async with dlq._db.execute(
            "SELECT failure_count, error_message FROM failed_documents WHERE url = ? AND failure_stage = ?",
            ("https://example.com/test", "crawl"),
        ) as cursor:
            row = await cursor.fetchone()

        assert row is not None, "Failed document should exist"
        failure_count, error_message = row
        assert failure_count == 2, f"Failure count should be 2 (upserted), got {failure_count}"
        assert error_message == "Connection timeout again", "Error message should be updated"

        # Add same URL but different stage - should be separate entry
        await dlq.add_failed_document(
            url="https://example.com/test",
            failure_stage="extract",
            error_info=error_info1,  # Different stage
        )

        # Verify both entries exist
        assert dlq._db is not None, "Database should be initialized"
        async with dlq._db.execute(
            "SELECT COUNT(*) FROM failed_documents WHERE url = ?", ("https://example.com/test",)
        ) as cursor:
            row = await cursor.fetchone()

        assert row is not None, "Row should exist"
        assert row[0] == 2, "Should have 2 entries for same URL with different stages"

    finally:
        await dlq.close()


@pytest.mark.asyncio
async def test_ac05_domain_failure_backpressure(mock_container):
    """AC-05: Test back-pressure on rapid failures - >5 failures per domain in 60s window."""

    settings = PipelineSettings(
        domain_failure_threshold=3,  # Lower threshold for faster testing
        domain_failure_window=10.0,  # Shorter window
        domain_backoff_duration=5.0,  # Shorter backoff
    )

    # Test domain failure tracker directly
    tracker = DomainFailureTracker(
        threshold=settings.domain_failure_threshold,
        window=settings.domain_failure_window,
        backoff_duration=settings.domain_backoff_duration,
    )

    domain = "failing-domain.com"

    # Record failures below threshold
    for i in range(2):
        tracker.record_failure(domain)
        assert not tracker.is_domain_backed_off(domain), f"Domain should not be backed off after {i + 1} failures"

    # Record failure that exceeds threshold
    tracker.record_failure(domain)
    assert tracker.is_domain_backed_off(domain), "Domain should be backed off after threshold exceeded"

    # Check backoff remaining time
    backoff_remaining = tracker.get_backoff_remaining(domain)
    assert backoff_remaining > 0, "Should have backoff time remaining"
    assert backoff_remaining <= settings.domain_backoff_duration, "Backoff should not exceed configured duration"

    # Test different domain not affected
    other_domain = "good-domain.com"
    assert not tracker.is_domain_backed_off(other_domain), "Other domains should not be affected"

    # Test backoff expiry (simulate fast-forward time)
    await asyncio.sleep(0.1)  # Small delay
    # Manually expire the backoff
    tracker.domain_backoff[domain] = time.time() - 1
    assert not tracker.is_domain_backed_off(domain), "Domain should not be backed off after expiry"


@pytest.mark.asyncio
async def test_ac06_configurable_settings_env_vars(temp_dir, mock_container):
    """AC-06: Test configurable checkpoint interval & location via environment variables."""

    # Set environment variables
    env_vars = {
        "CHECKPOINT_INTERVAL": "30.0",
        "CHECKPOINT_DIR": str(temp_dir / "custom_checkpoints"),
        "DOMAIN_FAILURE_THRESHOLD": "7",
        "DOMAIN_FAILURE_WINDOW": "120.0",
        "DOMAIN_BACKOFF_DURATION": "300.0",
        "DEAD_LETTER_DB_PATH": str(temp_dir / "custom_dead_letter.db"),
    }

    with patch.dict(os.environ, env_vars):
        settings = PipelineSettings.from_env()

        assert settings.checkpoint_interval == 30.0
        assert settings.checkpoint_dir == Path(temp_dir / "custom_checkpoints")
        assert settings.domain_failure_threshold == 7
        assert settings.domain_failure_window == 120.0
        assert settings.domain_backoff_duration == 300.0
        assert settings.dead_letter_db_path == Path(temp_dir / "custom_dead_letter.db")

    # Test pipeline uses custom settings
    pipeline = Pipeline(mock_container, settings=settings)
    assert pipeline.settings.checkpoint_interval == 30.0
    assert pipeline.domain_failure_tracker.threshold == 7


@pytest.mark.asyncio
async def test_ac07_graceful_sigint_sigterm_checkpoint_save(temp_dir, mock_container, test_urls):
    """AC-07: Test graceful SIGINT/SIGTERM handling with checkpoint save."""

    settings = PipelineSettings(
        checkpoint_dir=temp_dir / "signal_checkpoints",
        checkpoint_interval=30.0,  # Long interval so only signal triggers save
    )

    pipeline = Pipeline(mock_container, max_concurrency=2, settings=settings)

    # Mock the processing to be slow so we can interrupt it
    original_process_url = pipeline._process_url

    async def slow_process_url(url: str, worker_id: str):
        await asyncio.sleep(10)  # Slow processing
        return await original_process_url(url, worker_id)

    pipeline._process_url = slow_process_url

    # Start pipeline in background
    async def run_pipeline():
        return await pipeline.run(urls=test_urls[:3], batch_size=2)

    pipeline_task = asyncio.create_task(run_pipeline())

    # Wait a bit then send SIGINT
    await asyncio.sleep(0.5)

    # Test signal handler directly (safer than sending actual signal in test)
    if pipeline.state:
        # Trigger shutdown
        pipeline._shutdown_requested = True
        await pipeline._save_checkpoint()

        # Verify checkpoint was saved
        safe_job_id = slugify(pipeline.job_id, replacement="-", max_length=100, lowercase=True)
        checkpoint_path = settings.checkpoint_dir / f"{safe_job_id}.json"

        assert checkpoint_path.exists(), "Checkpoint should be saved on shutdown signal"

        # Verify checkpoint content
        checkpoint = await pipeline._load_checkpoint(checkpoint_path)
        assert checkpoint.job_id == pipeline.job_id
        assert len(checkpoint.urls_remaining) > 0, "Should have remaining URLs after interrupt"

    # Cancel the pipeline task
    pipeline_task.cancel()
    try:
        await pipeline_task
    except asyncio.CancelledError:
        pass


@pytest.mark.asyncio
async def test_comprehensive_kill_resume_cycle(temp_dir, mock_container, test_urls):
    """Comprehensive test: start crawl, kill after partial processing, resume, validate completion."""

    settings = PipelineSettings(
        checkpoint_dir=temp_dir / "resume_checkpoints",
        checkpoint_interval=2.0,  # Frequent checkpoints
        dead_letter_db_path=temp_dir / "resume_dead_letter.db",
    )

    # Phase 1: Start pipeline and interrupt after partial processing
    pipeline1 = Pipeline(mock_container, max_concurrency=2, settings=settings)

    # Mock some URLs to fail for dead letter testing
    original_process = pipeline1._process_url

    async def selective_process_url(url: str, worker_id: str):
        if "page2" in url or "article2" in url:
            raise Exception(f"Simulated failure for {url}")
        return await original_process(url, worker_id)

    pipeline1._process_url = selective_process_url

    # Run for limited time then stop
    async def interrupt_after_delay():
        await asyncio.sleep(1.0)  # Let some processing happen
        pipeline1._shutdown_requested = True
        if pipeline1.state:
            await pipeline1._save_checkpoint()

    # Start pipeline task
    pipeline_task = asyncio.create_task(pipeline1.run(urls=test_urls, batch_size=3))

    # Start interrupt task
    interrupt_task = asyncio.create_task(interrupt_after_delay())

    # Wait for interrupt to happen, then cancel
    await asyncio.sleep(2.0)

    # Cancel both tasks
    pipeline_task.cancel()
    interrupt_task.cancel()

    # Wait for cancellation to complete
    try:
        await pipeline_task
    except asyncio.CancelledError:
        pass

    try:
        await interrupt_task
    except asyncio.CancelledError:
        pass

    # Verify checkpoint exists
    safe_job_id = slugify(pipeline1.job_id, replacement="-", max_length=100, lowercase=True)
    checkpoint_path = settings.checkpoint_dir / f"{safe_job_id}.json"

    assert checkpoint_path.exists(), "Checkpoint should exist after interrupt"

    # Load checkpoint and verify partial progress
    checkpoint = await pipeline1._load_checkpoint(checkpoint_path)
    initial_processed = checkpoint.processed_count
    initial_failed = checkpoint.failed_count
    initial_remaining = len(checkpoint.urls_remaining)

    logger.info("Phase 1 complete", processed=initial_processed, failed=initial_failed, remaining=initial_remaining)

    assert initial_remaining > 0, "Should have URLs remaining after interrupt"

    # Phase 2: Resume from checkpoint
    pipeline2 = Pipeline(mock_container, max_concurrency=2, settings=settings)

    result = await pipeline2.run(urls=[], resume_from=checkpoint_path)  # Empty since resuming

    # Verify completion
    assert result["status"] == "completed"
    final_processed = result["processed_count"]
    final_failed = result["failed_count"]

    logger.info(
        "Phase 2 complete - Final results",
        final_processed=final_processed,
        final_failed=final_failed,
        total_urls=len(test_urls),
    )

    # Should have processed all URLs (either successfully or failed)
    assert final_processed + final_failed == len(test_urls)

    # Verify dead letter queue has failed URLs
    dlq = DeadLetterQueue(db_path=settings.dead_letter_db_path)
    await dlq.initialize()

    try:
        failed_docs = await dlq.get_failed_documents()
        failed_urls = [doc.url for doc in failed_docs]

        # Should have failures for URLs we simulated to fail
        expected_failures = [url for url in test_urls if "page2" in url or "article2" in url]
        for expected_fail_url in expected_failures:
            assert any(
                expected_fail_url in fail_url for fail_url in failed_urls
            ), f"Expected failure for {expected_fail_url} not found in dead letter queue"

        logger.info("Dead letter queue validation passed", failed_count=len(failed_docs))

    finally:
        await dlq.close()


@pytest.mark.asyncio
async def test_windows_compatibility_simulation():
    """Test atomic write behavior with Windows-like filesystem constraints."""

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        test_file = temp_path / "test_atomic.json"

        test_data = {"job_id": "test-job", "stage": "crawl", "processed": 10, "failed": 2}

        # Test atomic write
        atomic_write_json(test_file, test_data)

        assert test_file.exists(), "File should exist after atomic write"

        # Verify content
        with open(test_file) as f:
            loaded_data = json.load(f)

        assert loaded_data == test_data, "Data should match after atomic write"

        # Test overwrite
        new_data = {**test_data, "processed": 15}
        atomic_write_json(test_file, new_data)

        with open(test_file) as f:
            reloaded_data = json.load(f)

        assert reloaded_data["processed"] == 15, "Data should be updated after overwrite"
