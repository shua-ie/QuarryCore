"""Unit tests for pipeline shutdown and signal handling."""

import asyncio
import json
import signal
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from quarrycore.container import DependencyContainer
from quarrycore.pipeline import Pipeline, PipelineSettings, PipelineStage, PipelineState
from quarrycore.utils.atomic import atomic_json_dump


@pytest.mark.asyncio
async def test_shutdown_event_set_on_signal():
    """Test that shutdown event is set when signal is received."""
    with tempfile.TemporaryDirectory() as tmpdir:
        settings = PipelineSettings(checkpoint_dir=Path(tmpdir) / "checkpoints", checkpoint_interval=60.0)

        container = DependencyContainer()
        pipeline = Pipeline(container, settings=settings)

        # Initialize state
        pipeline.state = PipelineState(
            pipeline_id="test-pipeline",
            stage=PipelineStage.CRAWL,
            processed_count=0,
            failed_count=0,
            start_time=0,
            last_checkpoint=0,
            urls_remaining=["http://example.com"],
            batch_size=1,
            error_count_by_stage={},
        )

        # Setup signal handlers
        pipeline._setup_signal_handlers()

        # Verify shutdown event exists
        assert pipeline._shutdown_event is not None
        assert not pipeline._shutdown_event.is_set()

        # Simulate signal
        with patch("asyncio.get_running_loop") as mock_loop:
            mock_loop_instance = MagicMock()
            mock_loop_instance.is_closed.return_value = False
            mock_loop_instance.create_task = MagicMock()
            mock_loop.return_value = mock_loop_instance

            # Trigger signal handler directly
            handler = signal.getsignal(signal.SIGINT)
            if callable(handler):
                handler(signal.SIGINT, None)

            # Check that shutdown was requested
            assert pipeline._shutdown_requested
            assert pipeline._shutdown_event.is_set()

            # Check that graceful shutdown was scheduled
            mock_loop_instance.create_task.assert_called_once()
            args = mock_loop_instance.create_task.call_args[0]
            assert args[0].__name__ == "_graceful_shutdown"

        # Cleanup
        pipeline._cleanup_signal_handlers()


@pytest.mark.asyncio
async def test_graceful_shutdown_saves_checkpoint():
    """Test that graceful shutdown saves checkpoint."""
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir) / "checkpoints"
        settings = PipelineSettings(checkpoint_dir=checkpoint_dir, checkpoint_interval=60.0)

        container = DependencyContainer()
        pipeline = Pipeline(container, settings=settings)

        # Initialize state
        pipeline.state = PipelineState(
            pipeline_id="test-pipeline",
            stage=PipelineStage.CRAWL,
            processed_count=5,
            failed_count=1,
            start_time=0,
            last_checkpoint=0,
            urls_remaining=["http://example.com", "http://example2.com"],
            batch_size=1,
            error_count_by_stage={},
        )
        pipeline.is_running = True
        pipeline.job_id = "test-job-123"

        # Perform graceful shutdown
        await pipeline._graceful_shutdown("test_signal")

        # Check that checkpoint was saved
        checkpoint_files = list(checkpoint_dir.glob("*.json"))
        assert len(checkpoint_files) == 1

        # Verify checkpoint content
        with open(checkpoint_files[0]) as f:
            checkpoint_data = json.load(f)

        assert checkpoint_data["job_id"] == "test-job-123"
        assert checkpoint_data["pipeline_id"] == "test-pipeline"
        assert checkpoint_data["processed_count"] == 5
        assert checkpoint_data["failed_count"] == 1
        assert checkpoint_data["urls_remaining"] == ["http://example.com", "http://example2.com"]

        # Check that running flag was cleared
        assert not pipeline.is_running


@pytest.mark.asyncio
async def test_atomic_json_dump_with_timeout():
    """Test atomic_json_dump respects timeout and cleans up temp files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        target_path = Path(tmpdir) / "test.json"
        test_data = {"key": "value", "number": 42}

        # Test successful write
        result = await atomic_json_dump(test_data, target_path, timeout=2.0)
        assert result is True

        # Verify file was written
        assert target_path.exists()
        with open(target_path) as f:
            loaded_data = json.load(f)
        assert loaded_data == test_data

        # Remove the file for next test
        target_path.unlink()

        # Test timeout scenario - mock wait_for to raise TimeoutError
        with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError):
            result = await atomic_json_dump(test_data, target_path, timeout=0.05)
            assert result is False

        # Directory should be empty after timeout
        assert not any(Path(tmpdir).iterdir()), "Directory should be empty after timeout"


@pytest.mark.asyncio
async def test_emergency_checkpoint_on_no_event_loop():
    """Test emergency checkpoint is created when no event loop is available."""
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir) / "checkpoints"
        settings = PipelineSettings(checkpoint_dir=checkpoint_dir, checkpoint_interval=60.0)

        container = DependencyContainer()
        pipeline = Pipeline(container, settings=settings)

        # Initialize state
        pipeline.state = PipelineState(
            pipeline_id="test-pipeline",
            stage=PipelineStage.CRAWL,
            processed_count=3,
            failed_count=0,
            start_time=0,
            last_checkpoint=0,
            urls_remaining=["http://example.com"],
            batch_size=1,
            error_count_by_stage={},
        )
        pipeline.job_id = "emergency-test"
        pipeline.is_running = True  # Need this for emergency checkpoint to be attempted

        # Setup signal handlers
        pipeline._setup_signal_handlers()

        # Simulate signal with no running event loop
        with patch("asyncio.get_running_loop", side_effect=RuntimeError("No running event loop")):
            # Trigger signal handler
            handler = signal.getsignal(signal.SIGINT)
            if callable(handler):
                handler(signal.SIGINT, None)

            # Check for emergency checkpoint
            emergency_files = list(checkpoint_dir.glob("emergency_*.json"))
            assert len(emergency_files) == 1

            # Verify emergency checkpoint content
            with open(emergency_files[0]) as f:
                checkpoint_data = json.load(f)

            assert checkpoint_data["job_id"] == "emergency-test"
            assert checkpoint_data["processed_count"] == 3

        # Cleanup
        pipeline._cleanup_signal_handlers()


@pytest.mark.asyncio
async def test_graceful_shutdown_cancels_tasks():
    """Test that graceful shutdown properly cancels in-flight tasks."""
    with tempfile.TemporaryDirectory() as tmpdir:
        settings = PipelineSettings(checkpoint_dir=Path(tmpdir) / "checkpoints", checkpoint_interval=60.0)

        container = DependencyContainer()
        pipeline = Pipeline(container, settings=settings)

        # Mock task group and processing queue
        pipeline.task_group = MagicMock()
        pipeline.processing_queue = AsyncMock()
        pipeline.processing_queue.put_nowait = MagicMock()

        # Mock dead letter queue
        pipeline.dead_letter_queue = AsyncMock()

        # Initialize state
        pipeline.state = PipelineState(
            pipeline_id="test-pipeline",
            stage=PipelineStage.CRAWL,
            processed_count=0,
            failed_count=0,
            start_time=0,
            last_checkpoint=0,
            urls_remaining=[],
            batch_size=1,
            error_count_by_stage={},
        )
        pipeline.is_running = True

        # Perform graceful shutdown
        await pipeline._graceful_shutdown("test")

        # Verify None was put in queue to signal workers
        assert pipeline.processing_queue.put_nowait.call_count > 0
        pipeline.processing_queue.put_nowait.assert_called_with(None)

        # Verify dead letter queue was closed
        pipeline.dead_letter_queue.close.assert_called_once()
