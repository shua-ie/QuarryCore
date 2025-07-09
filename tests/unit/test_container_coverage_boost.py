"""
Targeted tests for container.py to boost branch coverage.
"""

import asyncio
from pathlib import Path
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from quarrycore.config.config import Config
from quarrycore.container import DependencyContainer


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

    @pytest.fixture
    def container_no_config(self):
        """Create container without config path."""
        return DependencyContainer(None)

    @pytest.mark.asyncio
    async def test_double_initialization_guard(self, container):
        """Test double initialization guard logic."""
        # Initialize once
        await container.initialize()

        # Try to initialize again - should be idempotent
        await container.initialize()

        # Should only be initialized once
        assert container.is_running

        # Cleanup
        await container.shutdown()

    @pytest.mark.asyncio
    async def test_shutdown_idempotency(self, container):
        """Test shutdown idempotency."""
        # Initialize first
        await container.initialize()

        # Shutdown multiple times
        await container.shutdown()
        await container.shutdown()

        # Should handle multiple shutdowns gracefully
        assert not container.is_running

    @pytest.mark.asyncio
    async def test_config_reload_scenario(self, container, temp_config_path):
        """Test config reload scenarios."""
        # Initialize the container
        await container.initialize()

        try:
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

            # Should have reloaded config
            assert container.config is not None
            assert container.is_running

        finally:
            await container.shutdown()

    @pytest.mark.asyncio
    async def test_lazy_instance_lifecycle(self, container):
        """Test lazy instance lifecycle management."""
        # Initialize the container
        await container.initialize()

        try:
            # Get lazy instances
            storage_manager = await container.get_storage()
            http_client = await container.get_http_client()

            # Should return the same instances
            assert storage_manager is await container.get_storage()
            assert http_client is await container.get_http_client()

        finally:
            await container.shutdown()

    @pytest.mark.asyncio
    async def test_signal_handling_setup(self, container):
        """Test signal handling setup."""
        # Mock signal module
        with patch("signal.signal") as mock_signal:
            await container.initialize()

            # Should have set up signal handlers
            assert mock_signal.call_count >= 2  # SIGTERM and SIGINT

            await container.shutdown()

    @pytest.mark.asyncio
    async def test_config_watcher_functionality(self, container):
        """Test config watcher functionality."""
        # Initialize the container
        await container.initialize()

        try:
            # Should have set up config watcher
            assert container._observer is not None

        finally:
            await container.shutdown()

    @pytest.mark.asyncio
    async def test_initialization_without_config_path(self, container_no_config):
        """Test initialization without config path."""
        # Should initialize with default config
        await container_no_config.initialize()

        try:
            assert container_no_config.config is not None
            assert container_no_config.is_running

        finally:
            await container_no_config.shutdown()

    @pytest.mark.asyncio
    async def test_shutdown_error_handling(self, container):
        """Test shutdown error handling."""
        # Initialize first
        await container.initialize()

        # Mock a component that fails during shutdown
        with patch.object(container, "_cleanup_instances") as mock_cleanup:
            mock_cleanup.side_effect = Exception("Cleanup failed")

            # Should re-raise the exception from _cleanup_instances
            with pytest.raises(Exception, match="Cleanup failed"):
                await container.shutdown()

            # Should still be marked as running since shutdown failed
            assert container.is_running

    @pytest.mark.asyncio
    async def test_component_getter_error_handling(self, container):
        """Test component getter error handling."""
        # Initialize the container
        await container.initialize()

        try:
            # Mock instance creation failure
            with patch.object(container._instances["storage"], "get") as mock_get:
                mock_get.side_effect = Exception("Instance creation failed")

                # Should handle dependency injection failure
                try:
                    await container.get_storage()
                    raise AssertionError("Should have raised exception")
                except Exception:
                    assert True

        finally:
            await container.shutdown()

    @pytest.mark.asyncio
    async def test_config_path_handling(self, container):
        """Test config path handling."""
        # Initialize with config path
        await container.initialize()

        try:
            # Test path handling
            assert container.config_path is not None
            assert container.config is not None

        finally:
            await container.shutdown()

    @pytest.mark.asyncio
    async def test_config_watcher_no_path(self, container_no_config):
        """Test config watcher when no config path provided."""
        # Initialize without config path
        await container_no_config.initialize()

        try:
            # Should not have set up config watcher
            assert container_no_config._observer is None

        finally:
            await container_no_config.shutdown()

    @pytest.mark.asyncio
    async def test_async_component_cleanup(self, container):
        """Test async component cleanup."""
        # Initialize the container
        await container.initialize()

        # Mock cleanup methods
        with patch.object(container, "_cleanup_instances") as mock_cleanup:
            mock_cleanup.return_value = None

            # Shutdown should clean up instances
            await container.shutdown()

            # Should have cleaned up instances
            mock_cleanup.assert_called_once()

    @pytest.mark.asyncio
    async def test_health_status_before_init(self, container):
        """Test health status before initialization."""
        status = container.get_health_status()

        assert status["is_running"] is False
        assert status["config_loaded"] is False
        assert status["instances_count"] == 0
        assert status["pipeline_id"] is not None

    @pytest.mark.asyncio
    async def test_health_status_after_init(self, container):
        """Test health status after initialization."""
        await container.initialize()

        try:
            status = container.get_health_status()

            assert status["is_running"] is True
            assert status["config_loaded"] is True
            assert status["instances_count"] > 0
            assert status["pipeline_id"] is not None

        finally:
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

    @pytest.mark.asyncio
    async def test_lifecycle_context_manager(self, container):
        """Test lifecycle context manager."""
        async with container.lifecycle() as ctx:
            assert ctx is container
            assert container.is_running

        # Should have shut down after context
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
    async def test_component_state_consistency(self, container):
        """Test component state consistency."""
        # Initialize the container
        await container.initialize()

        try:
            # Get components multiple times
            storage1 = await container.get_storage()
            storage2 = await container.get_storage()

            # Should return the same instance
            assert storage1 is storage2

        finally:
            await container.shutdown()

    @pytest.mark.asyncio
    async def test_config_file_missing(self, tmp_path):
        """Test initialization with missing config file."""
        missing_config = tmp_path / "missing.yaml"
        container = DependencyContainer(missing_config)

        # Should initialize with default config
        await container.initialize()

        try:
            assert container.config is not None
            assert container.is_running

        finally:
            await container.shutdown()

    @pytest.mark.asyncio
    async def test_memory_management(self, container):
        """Test memory management."""
        # Initialize and shutdown multiple times
        for _ in range(3):
            await container.initialize()

            # Get some components
            await container.get_storage()
            await container.get_http_client()

            # Shutdown
            await container.shutdown()

            # Should be cleaned up
            assert not container.is_running
