"""
Tests for DependencyContainer covering edge cases and error paths.
"""

import asyncio
import os
import signal
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from quarrycore.config import Config
from quarrycore.container import DependencyContainer


class TestContainerShutdownPaths:
    """Test shutdown-related code paths in DependencyContainer."""

    @pytest.mark.asyncio
    async def test_duplicate_shutdown_calls_are_idempotent(self):
        """Test that calling shutdown multiple times is safe and idempotent."""
        container = DependencyContainer()

        # Initialize container
        await container.initialize()
        assert container.is_running is True

        # First shutdown should work normally
        await container.shutdown()
        assert container.is_running is False

        # Second shutdown should be idempotent (no errors)
        await container.shutdown()
        assert container.is_running is False

        # Third shutdown should also be idempotent
        await container.shutdown()
        assert container.is_running is False

    @pytest.mark.asyncio
    async def test_shutdown_handler_execution_and_error_isolation(self):
        """Test that shutdown handlers execute properly and errors are isolated."""
        container = DependencyContainer()

        # Track handler execution
        handler_calls = []

        # Add sync handler that succeeds
        def sync_handler_success():
            handler_calls.append("sync_success")

        # Add sync handler that fails
        def sync_handler_fail():
            handler_calls.append("sync_fail")
            raise Exception("Sync handler error")

        # Add async handler that succeeds
        async def async_handler_success():
            handler_calls.append("async_success")

        # Add async handler that fails
        async def async_handler_fail():
            handler_calls.append("async_fail")
            raise Exception("Async handler error")

        # Add all handlers
        container.add_shutdown_handler(sync_handler_success)
        container.add_shutdown_handler(sync_handler_fail)
        container.add_shutdown_handler(async_handler_success)
        container.add_shutdown_handler(async_handler_fail)

        # Initialize and shutdown
        await container.initialize()
        await container.shutdown()

        # Verify all handlers were called despite errors
        assert "sync_success" in handler_calls
        assert "sync_fail" in handler_calls
        assert "async_success" in handler_calls
        assert "async_fail" in handler_calls

        # Verify container is properly shut down
        assert container.is_running is False

    @pytest.mark.asyncio
    async def test_cleanup_instances_error_handling(self):
        """Test that instance cleanup errors are properly handled."""
        container = DependencyContainer()

        # Mock instances that will fail during cleanup
        mock_instance_good = AsyncMock()
        mock_instance_bad = AsyncMock()
        mock_instance_bad.cleanup.side_effect = Exception("Cleanup error")

        # Manually add instances to test cleanup
        container._instances = {"good": mock_instance_good, "bad": mock_instance_bad}

        await container.initialize()

        # Should not raise exception despite cleanup failure
        await container.shutdown()

        # Verify cleanup was attempted on both
        mock_instance_good.cleanup.assert_called_once()
        mock_instance_bad.cleanup.assert_called_once()


class TestContainerSignalHandling:
    """Test signal handling in DependencyContainer."""

    @pytest.mark.asyncio
    async def test_signal_handler_registration(self):
        """Test that signal handlers are properly registered."""
        container = DependencyContainer()

        # Mock signal.signal to track calls
        with patch("signal.signal") as mock_signal:
            await container.initialize()

            # Verify SIGTERM and SIGINT handlers were registered
            calls = mock_signal.call_args_list
            signal_nums = [call[0][0] for call in calls]

            assert signal.SIGTERM in signal_nums
            assert signal.SIGINT in signal_nums

        await container.shutdown()

    @pytest.mark.asyncio
    async def test_signal_handler_triggers_shutdown(self):
        """Test that signal handlers trigger shutdown properly."""
        container = DependencyContainer()

        # Track shutdown calls
        shutdown_called = False
        original_shutdown = container.shutdown

        async def track_shutdown():
            nonlocal shutdown_called
            shutdown_called = True
            await original_shutdown()

        container.shutdown = track_shutdown

        # Initialize container
        await container.initialize()

        # Get the signal handler that was registered
        signal_handler = None
        with patch("signal.signal") as mock_signal:
            await container._setup_signal_handlers()
            # Get the handler function from the first call
            if mock_signal.call_args_list:
                signal_handler = mock_signal.call_args_list[0][0][1]

        # Simulate signal reception
        if signal_handler:
            signal_handler(signal.SIGTERM, None)
            # Give async task time to execute
            await asyncio.sleep(0.1)

        # Should have triggered shutdown
        assert shutdown_called is True

    @pytest.mark.asyncio
    async def test_signal_handler_with_different_signals(self):
        """Test signal handlers work with both SIGTERM and SIGINT."""
        container = DependencyContainer()

        shutdown_signals = []

        # Mock shutdown to track which signals trigger it
        async def mock_shutdown():
            shutdown_signals.append("shutdown")

        container.shutdown = mock_shutdown

        await container.initialize()

        # Get signal handler
        signal_handler = None
        with patch("signal.signal") as mock_signal:
            await container._setup_signal_handlers()
            if mock_signal.call_args_list:
                signal_handler = mock_signal.call_args_list[0][0][1]

        if signal_handler:
            # Test SIGTERM
            signal_handler(signal.SIGTERM, None)
            await asyncio.sleep(0.1)

            # Test SIGINT
            signal_handler(signal.SIGINT, None)
            await asyncio.sleep(0.1)

        # Both signals should have triggered shutdown
        assert len(shutdown_signals) == 2


class TestContainerConfigReload:
    """Test configuration reloading in DependencyContainer."""

    @pytest.mark.asyncio
    async def test_reload_config_with_file_changes(self):
        """Test configuration reloading when file changes."""
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(
                """
monitoring:
  enabled: true
  metrics_port: 9090
storage:
  db_path: "test.db"
"""
            )
            config_path = Path(f.name)

        try:
            container = DependencyContainer(config_path)
            await container.initialize()

            original_config = container.config

            # Modify config file
            with open(config_path, "w") as f:
                f.write(
                    """
monitoring:
  enabled: false
  prometheus_port: 9091
storage:
  db_path: "test2.db"
"""
                )

            # Reload config
            await container.reload_config()

            # Verify config changed
            assert container.config != original_config
            assert container.config is not None
            assert container.config.monitoring.enabled is False
            assert container.config.monitoring.prometheus_port == 9091

            await container.shutdown()

        finally:
            # Clean up
            if config_path.exists():
                config_path.unlink()

    @pytest.mark.asyncio
    async def test_reload_config_without_file(self):
        """Test configuration reloading when no config file exists."""
        container = DependencyContainer()
        await container.initialize()

        # Reload should use default config

    @pytest.mark.asyncio
    async def test_reload_config_creates_new_instances(self):
        """Test that config reload creates new instances."""
        container = DependencyContainer()
        await container.initialize()

        # Get reference to original instances
        original_instances = dict(container._instances)

        # Reload config
        await container.reload_config()

        # Instances should be recreated
        assert container._instances != original_instances
        assert len(container._instances) > 0

        await container.shutdown()

    @pytest.mark.asyncio
    async def test_config_watcher_file_modification(self):
        """Test that config watcher detects file modifications."""
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("monitoring:\n  enabled: true\n")
            config_path = Path(f.name)

        try:
            container = DependencyContainer(config_path)

            # Mock reload_config to track calls
            reload_called = False

            async def mock_reload():
                nonlocal reload_called
                reload_called = True

            container.reload_config = mock_reload

            await container.initialize()

            # Simulate file modification event
            from quarrycore.container import ConfigWatcher

            watcher = ConfigWatcher(container)

            # Create mock file system event
            mock_event = MagicMock()
            mock_event.is_directory = False
            mock_event.src_path = str(config_path)

            # Trigger file modification
            watcher.on_modified(mock_event)

            # Give async task time to execute
            await asyncio.sleep(0.1)

            # Should have triggered reload
            assert reload_called is True

            await container.shutdown()

        finally:
            # Clean up
            if config_path.exists():
                config_path.unlink()


class TestContainerLifecycle:
    """Test container lifecycle management."""

    @pytest.mark.asyncio
    async def test_lifecycle_context_manager(self):
        """Test container lifecycle context manager."""
        container = DependencyContainer()

        # Test successful lifecycle
        async with container.lifecycle() as ctx:
            assert ctx is container
            assert container.is_running is True

        # Should be shut down after context
        assert container.is_running is False

    @pytest.mark.asyncio
    async def test_lifecycle_context_manager_with_exception(self):
        """Test container lifecycle handles exceptions properly."""
        container = DependencyContainer()

        # Test lifecycle with exception
        with pytest.raises(ValueError):
            async with container.lifecycle():
                assert container.is_running is True
                raise ValueError("Test exception")

        # Should still be shut down after exception
        assert container.is_running is False

    @pytest.mark.asyncio
    async def test_health_status_reporting(self):
        """Test health status reporting."""
        container = DependencyContainer()

        # Before initialization
        status = container.get_health_status()
        assert status["is_running"] is False
        assert status["config_loaded"] is False
        assert status["pipeline_id"] is not None

        # After initialization
        await container.initialize()
        status = container.get_health_status()
        assert status["is_running"] is True
        assert status["config_loaded"] is True
        assert status["instances_count"] > 0

        await container.shutdown()
        status = container.get_health_status()
        assert status["is_running"] is False

    @pytest.mark.asyncio
    async def test_initialize_with_existing_config(self):
        """Test initialization with pre-existing config."""
        config = Config()
        container = DependencyContainer()
        container.config = config

        await container.initialize()

        # Should use existing config
        assert container.config is config
        assert container.is_running is True

        await container.shutdown()

    @pytest.mark.asyncio
    async def test_observer_cleanup_on_shutdown(self):
        """Test that file observer is properly cleaned up on shutdown."""
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("monitoring:\n  enabled: true\n")
            config_path = Path(f.name)

        try:
            container = DependencyContainer(config_path)
            await container.initialize()

            # Should have observer
            assert container._observer is not None

            # Mock observer to track cleanup calls
            mock_observer = MagicMock()
            container._observer = mock_observer

            await container.shutdown()

            # Observer should be stopped and joined
            mock_observer.stop.assert_called_once()
            mock_observer.join.assert_called_once()

        finally:
            # Clean up
            if config_path.exists():
                config_path.unlink()
