"""
Targeted tests for container.py to boost branch coverage.
"""

import asyncio
from pathlib import Path
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from uuid import uuid4

import pytest
from quarrycore.config import Config
from quarrycore.container import ConfigWatcher, DependencyContainer, LazyInstance


class TestContainerCoverageBoost:
    """Targeted tests to boost container.py branch coverage."""

    @pytest.fixture
    def temp_config_path(self, tmp_path):
        """Create temporary config file."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            """
monitoring:
  enabled: true
storage:
  hot:
    db_path: ./data/test.db
"""
        )
        return config_path

    @pytest.fixture
    def container(self, temp_config_path):
        """Create container instance."""
        return DependencyContainer(temp_config_path)

    @pytest.mark.asyncio
    async def test_double_initialization_guard(self, container):
        """Test double initialization guard."""
        # First initialization should succeed
        await container.initialize()
        assert container.is_running

        # Second initialization should be idempotent
        await container.initialize()
        assert container.is_running

        await container.shutdown()

    @pytest.mark.asyncio
    async def test_initialize_without_config_path(self):
        """Test initialization without config path."""
        container = DependencyContainer(config_path=None)

        # Should initialize with default config
        await container.initialize()
        assert container.config is not None
        assert container.is_running

        await container.shutdown()

    @pytest.mark.asyncio
    async def test_initialize_with_nonexistent_config_path(self):
        """Test initialization with nonexistent config path."""
        nonexistent_path = Path("/nonexistent/config.yaml")
        container = DependencyContainer(config_path=nonexistent_path)

        # Should initialize with default config
        await container.initialize()
        assert container.config is not None
        assert container.is_running

        await container.shutdown()

    @pytest.mark.asyncio
    async def test_reload_config_with_changes(self, container, temp_config_path):
        """Test config reload with actual changes."""
        await container.initialize()
        original_config = container.config

        # Modify config file
        temp_config_path.write_text(
            """
monitoring:
  enabled: false
storage:
  hot:
    db_path: ./data/different.db
"""
        )

        # Reload config
        await container.reload_config()

        # Config should be different
        assert container.config != original_config
        await container.shutdown()

    @pytest.mark.asyncio
    async def test_reload_config_failure_handling(self, container, temp_config_path):
        """Test config reload failure handling."""
        await container.initialize()

        # Create invalid config file
        temp_config_path.write_text("invalid: yaml: content: [")

        # Config reload should handle error gracefully
        await container.reload_config()

        # Container should still be running
        assert container.is_running
        await container.shutdown()

    @pytest.mark.asyncio
    async def test_shutdown_idempotency(self, container):
        """Test shutdown idempotency."""
        await container.initialize()

        # First shutdown should work
        await container.shutdown()
        assert not container.is_running

        # Second shutdown should be idempotent
        await container.shutdown()
        assert not container.is_running

    @pytest.mark.asyncio
    async def test_shutdown_without_initialization(self, container):
        """Test shutdown without initialization."""
        # Should handle shutdown gracefully even without init
        await container.shutdown()
        assert not container.is_running

    @pytest.mark.asyncio
    async def test_lifecycle_context_manager_exception(self, container):
        """Test lifecycle context manager with exception."""
        try:
            async with container.lifecycle():
                assert container.is_running
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Should have shut down despite exception
        assert not container.is_running

    @pytest.mark.asyncio
    async def test_cleanup_instances_with_errors(self, container):
        """Test cleanup with instance errors."""
        await container.initialize()

        # Mock an instance that throws error on cleanup
        mock_instance = Mock()
        mock_instance.cleanup.side_effect = Exception("Cleanup failed")

        container._instances["test"] = mock_instance

        # Should handle cleanup errors gracefully
        await container._cleanup_instances()

        await container.shutdown()

    @pytest.mark.asyncio
    async def test_lazy_instance_initialization_failure(self):
        """Test lazy instance initialization failure."""

        def failing_factory():
            raise ValueError("Factory failed")

        lazy_instance = LazyInstance(failing_factory)

        with pytest.raises(ValueError, match="Factory failed"):
            await lazy_instance.get()

    @pytest.mark.asyncio
    async def test_lazy_instance_with_async_initialize(self):
        """Test lazy instance with async initialize method."""

        class MockComponent:
            def __init__(self):
                self.initialized = False

            async def initialize(self):
                self.initialized = True

        lazy_instance = LazyInstance(MockComponent)
        component = await lazy_instance.get()

        assert component.initialized

        # Second get should return same instance
        component2 = await lazy_instance.get()
        assert component is component2

    @pytest.mark.asyncio
    async def test_lazy_instance_cleanup_with_close(self):
        """Test lazy instance cleanup with close method."""

        class MockComponent:
            def __init__(self):
                self.closed = False

            async def close(self):
                self.closed = True

        lazy_instance = LazyInstance(MockComponent)
        component = await lazy_instance.get()

        await lazy_instance.cleanup()
        assert component.closed

    @pytest.mark.asyncio
    async def test_lazy_instance_cleanup_without_close(self):
        """Test lazy instance cleanup without close method."""

        class MockComponent:
            def __init__(self):
                pass

        lazy_instance = LazyInstance(MockComponent)
        await lazy_instance.get()

        # Should not raise error
        await lazy_instance.cleanup()

    @pytest.mark.asyncio
    async def test_get_instance_with_lock_contention(self, container):
        """Test getting instance with lock contention."""
        await container.initialize()

        # Simulate concurrent access
        async def get_storage():
            return await container.get_storage()

        # Multiple concurrent calls should work
        tasks = [get_storage() for _ in range(5)]
        results = await asyncio.gather(*tasks)

        # All should return the same instance
        assert all(results[0] is result for result in results)

        await container.shutdown()

    @pytest.mark.asyncio
    async def test_signal_handler_setup_branch(self, container):
        """Test signal handler setup branch."""
        with patch("signal.signal") as mock_signal:
            await container.initialize()

            # Should have set up signal handlers
            assert mock_signal.call_count >= 2  # SIGTERM and SIGINT

            await container.shutdown()

    @pytest.mark.asyncio
    async def test_config_watcher_setup_branch(self, container, temp_config_path):
        """Test config watcher setup branch."""
        with patch("quarrycore.container.Observer") as mock_observer:
            mock_observer_instance = Mock()
            mock_observer.return_value = mock_observer_instance

            await container.initialize()

            # Should have set up config watcher
            assert mock_observer.called
            assert mock_observer_instance.schedule.called
            assert mock_observer_instance.start.called

            await container.shutdown()

    @pytest.mark.asyncio
    async def test_config_watcher_without_config_path(self):
        """Test config watcher without config path."""
        container = DependencyContainer(config_path=None)

        with patch("quarrycore.container.Observer") as mock_observer:
            await container.initialize()

            # Should not set up config watcher
            assert not mock_observer.called

            await container.shutdown()

    @pytest.mark.asyncio
    async def test_shutdown_handlers_execution(self, container):
        """Test shutdown handlers execution."""
        await container.initialize()

        handler_called = False

        def test_handler():
            nonlocal handler_called
            handler_called = True

        container.add_shutdown_handler(test_handler)

        await container.shutdown()

        assert handler_called

    @pytest.mark.asyncio
    async def test_shutdown_handlers_with_errors(self, container):
        """Test shutdown handlers with errors."""
        await container.initialize()

        def failing_handler():
            raise ValueError("Handler failed")

        def working_handler():
            pass

        container.add_shutdown_handler(failing_handler)
        container.add_shutdown_handler(working_handler)

        # Should handle errors gracefully
        await container.shutdown()

    @pytest.mark.asyncio
    async def test_shutdown_handlers_async(self, container):
        """Test async shutdown handlers."""
        await container.initialize()

        handler_called = False

        async def async_handler():
            nonlocal handler_called
            handler_called = True

        container.add_shutdown_handler(async_handler)

        await container.shutdown()

        assert handler_called

    def test_get_health_status_before_init(self, container):
        """Test health status before initialization."""
        status = container.get_health_status()

        assert status["is_running"] is False
        assert status["config_loaded"] is False
        assert status["instances_count"] == 0
        assert status["pipeline_id"] is not None

    @pytest.mark.asyncio
    async def test_get_health_status_after_init(self, container):
        """Test health status after initialization."""
        await container.initialize()

        status = container.get_health_status()

        assert status["is_running"] is True
        assert status["config_loaded"] is True
        assert status["instances_count"] > 0
        assert status["pipeline_id"] is not None

        await container.shutdown()

    def test_config_watcher_file_modification(self, container):
        """Test config watcher file modification handling."""
        watcher = ConfigWatcher(container)

        # Create mock file event
        mock_event = Mock()
        mock_event.is_directory = False
        mock_event.src_path = "/path/to/config.yaml"

        with patch.object(container, "reload_config"):
            with patch("asyncio.create_task") as mock_create_task:
                watcher.on_modified(mock_event)

                # Should have triggered reload
                assert mock_create_task.called

    def test_config_watcher_directory_modification(self, container):
        """Test config watcher directory modification handling."""
        watcher = ConfigWatcher(container)

        # Create mock directory event
        mock_event = Mock()
        mock_event.is_directory = True
        mock_event.src_path = "/path/to/directory"

        with patch.object(container, "reload_config"):
            with patch("asyncio.create_task") as mock_create_task:
                watcher.on_modified(mock_event)

                # Should not have triggered reload for directory
                assert not mock_create_task.called

    def test_config_watcher_non_yaml_modification(self, container):
        """Test config watcher non-yaml file modification handling."""
        watcher = ConfigWatcher(container)

        # Create mock non-yaml file event
        mock_event = Mock()
        mock_event.is_directory = False
        mock_event.src_path = "/path/to/file.txt"

        with patch.object(container, "reload_config"):
            with patch("asyncio.create_task") as mock_create_task:
                watcher.on_modified(mock_event)

                # Should not have triggered reload for non-yaml
                assert not mock_create_task.called
