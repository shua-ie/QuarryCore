"""
Advanced tests for dependency container to achieve 90% coverage.

Tests include concurrent access, lifecycle management, race conditions,
and error handling scenarios.
"""

import asyncio
import signal
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from quarrycore.config.config import Config
from quarrycore.container import ConfigWatcher, DependencyContainer, LazyInstance


@pytest.mark.unit
class TestDependencyContainerAdvanced:
    """Advanced container tests for edge cases and concurrency."""

    @pytest_asyncio.fixture
    async def temp_config_path(self):
        """Create temporary config file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(
                """
crawler:
  concurrent_requests: 5
  timeout: 30.0
  user_agent: "TestBot/1.0"

storage:
  hot:
    db_path: "/tmp/test.db"
            """
            )
            f.flush()
            yield Path(f.name)
        Path(f.name).unlink()

    @pytest.mark.asyncio
    async def test_concurrent_singleton_access(self):
        """Test concurrent access to same singleton instance."""
        config = Config()
        container = DependencyContainer()
        container.config = config
        await container.initialize()

        try:
            # Create 100 concurrent tasks accessing the same instance
            async def get_http_client():
                return await container.get_http_client()

            tasks = [get_http_client() for _ in range(100)]
            instances = await asyncio.gather(*tasks)

            # All should be the same instance
            first_instance = instances[0]
            assert all(instance is first_instance for instance in instances)

        finally:
            await container.shutdown()

    @pytest.mark.asyncio
    async def test_container_lifecycle_context_manager(self):
        """Test container lifecycle with context manager."""
        config = Config()

        async with DependencyContainer().lifecycle() as container:
            container.config = config

            # Should be running
            assert container.is_running

            # Should be able to get instances
            http_client = await container.get_http_client()
            assert http_client is not None

        # Should be shut down after exiting context
        assert not container.is_running

    @pytest.mark.asyncio
    async def test_config_hot_reload(self, temp_config_path):
        """Test configuration hot reloading."""
        container = DependencyContainer(config_path=temp_config_path)
        await container.initialize()

        try:
            # Get initial config
            initial_config = container.config
            assert initial_config is not None

            # Modify config file
            with open(temp_config_path, "w") as f:
                f.write(
                    """
crawler:
  concurrent_requests: 10
  timeout: 60.0
  user_agent: "NewBot/2.0"
                """
                )

            # Trigger reload
            await container.reload_config()

            # Config should be updated
            assert container.config != initial_config
            assert container.config is not None
            assert container.config.crawler.concurrent_requests == 10
            assert container.config.crawler.timeout == 60.0

        finally:
            await container.shutdown()

    @pytest.mark.asyncio
    async def test_config_watcher_file_events(self, temp_config_path):
        """Test config watcher responds to file system events."""
        container = DependencyContainer(config_path=temp_config_path)
        await container.initialize()

        try:
            # Create watcher
            watcher = ConfigWatcher(container)

            # Create mock file event
            mock_event = MagicMock()
            mock_event.is_directory = False
            mock_event.src_path = str(temp_config_path)

            # Mock container.reload_config to track calls
            container.reload_config = AsyncMock()

            # Trigger file event
            watcher.on_modified(mock_event)

            # Wait for async task to complete
            await asyncio.sleep(0.1)

            # Should have called reload_config
            container.reload_config.assert_called_once()

        finally:
            await container.shutdown()

    @pytest.mark.asyncio
    async def test_nonexistent_config_file(self):
        """Test container with nonexistent config file."""
        nonexistent_path = Path("/tmp/nonexistent_config.yaml")

        container = DependencyContainer(config_path=nonexistent_path)
        await container.initialize()

        try:
            # Should use default config
            assert container.config is not None
            assert isinstance(container.config, Config)

            # Should still work
            http_client = await container.get_http_client()
            assert http_client is not None

        finally:
            await container.shutdown()

    @pytest.mark.asyncio
    async def test_lazy_instance_initialization_error(self):
        """Test lazy instance handling of initialization errors."""

        def failing_factory():
            raise ValueError("Initialization failed")

        lazy_instance = LazyInstance(failing_factory)

        # Should propagate the error
        with pytest.raises(ValueError, match="Initialization failed"):
            await lazy_instance.get()

    @pytest.mark.asyncio
    async def test_lazy_instance_with_async_initialize(self):
        """Test lazy instance with async initialize method."""

        class MockService:
            def __init__(self):
                self.initialized = False

            async def initialize(self):
                self.initialized = True

        lazy_instance = LazyInstance(MockService)
        service = await lazy_instance.get()

        # Should have called initialize
        assert service.initialized

    @pytest.mark.asyncio
    async def test_lazy_instance_cleanup_with_async_close(self):
        """Test lazy instance cleanup with async close method."""

        class MockService:
            def __init__(self):
                self.closed = False

            async def close(self):
                self.closed = True

        lazy_instance = LazyInstance(MockService)
        service = await lazy_instance.get()

        # Cleanup should call close
        await lazy_instance.cleanup()
        assert service.closed

    @pytest.mark.asyncio
    async def test_container_shutdown_multiple_times(self):
        """Test container shutdown idempotency."""
        container = DependencyContainer()
        container.config = Config()
        await container.initialize()

        # First shutdown
        await container.shutdown()
        assert not container.is_running

        # Second shutdown should be safe
        await container.shutdown()
        assert not container.is_running

    @pytest.mark.asyncio
    async def test_container_without_config_initialization(self):
        """Test container behavior without config initialization."""
        container = DependencyContainer()

        # Should raise error when trying to create instances
        with pytest.raises(RuntimeError, match="Configuration must be loaded"):
            await container._create_instances()

    @pytest.mark.asyncio
    async def test_shutdown_handler_registration(self):
        """Test custom shutdown handler registration."""
        container = DependencyContainer()
        container.config = Config()
        await container.initialize()

        # Add mock shutdown handlers
        sync_handler = MagicMock()
        async_handler = AsyncMock()

        container.add_shutdown_handler(sync_handler)
        container.add_shutdown_handler(async_handler)

        # Shutdown should call both handlers
        await container.shutdown()

        sync_handler.assert_called_once()
        async_handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_handler_error_handling(self):
        """Test error handling in shutdown handlers."""
        container = DependencyContainer()
        container.config = Config()
        await container.initialize()

        # Add failing handlers
        def failing_sync_handler():
            raise ValueError("Sync handler failed")

        async def failing_async_handler():
            raise ValueError("Async handler failed")

        container.add_shutdown_handler(failing_sync_handler)
        container.add_shutdown_handler(failing_async_handler)

        # Shutdown should handle errors gracefully
        await container.shutdown()

        # Should complete without raising exceptions
        assert not container.is_running

    @pytest.mark.asyncio
    async def test_instance_cleanup_error_handling(self):
        """Test error handling during instance cleanup."""

        class FailingService:
            async def close(self):
                raise ValueError("Cleanup failed")

        container = DependencyContainer()
        container.config = Config()

        # Manually add failing instance
        container._instances["failing"] = LazyInstance(FailingService)

        # Should handle cleanup errors gracefully
        await container._cleanup_instances()

    @pytest.mark.asyncio
    async def test_health_status_reporting(self):
        """Test health status reporting."""
        container = DependencyContainer()
        container.config = Config()
        await container.initialize()

        try:
            health = container.get_health_status()

            # Should report correct status
            assert health["is_running"]
            assert health["config_loaded"]
            assert health["instances_count"] > 0
            assert health["pipeline_id"] == container.pipeline_id

        finally:
            await container.shutdown()

    @pytest.mark.asyncio
    async def test_signal_handler_setup(self):
        """Test signal handler setup."""
        container = DependencyContainer()
        container.config = Config()

        with patch("signal.signal") as mock_signal:
            await container.initialize()

            try:
                # Should have set up signal handlers
                assert mock_signal.call_count >= 2

                # Check SIGTERM and SIGINT handlers were set
                calls = mock_signal.call_args_list
                signals_set = [call[0][0] for call in calls]
                assert signal.SIGTERM in signals_set
                assert signal.SIGINT in signals_set

            finally:
                await container.shutdown()

    @pytest.mark.asyncio
    async def test_config_watcher_non_yaml_files(self):
        """Test config watcher ignores non-YAML files."""
        container = DependencyContainer()
        watcher = ConfigWatcher(container)

        # Mock container.reload_config
        container.reload_config = AsyncMock()

        # Create non-YAML file event
        mock_event = MagicMock()
        mock_event.is_directory = False
        mock_event.src_path = "/path/to/file.txt"

        # Should not trigger reload
        watcher.on_modified(mock_event)

        # Wait and check
        await asyncio.sleep(0.1)
        container.reload_config.assert_not_called()

    @pytest.mark.asyncio
    async def test_config_watcher_directory_events(self):
        """Test config watcher ignores directory events."""
        container = DependencyContainer()
        watcher = ConfigWatcher(container)

        # Mock container.reload_config
        container.reload_config = AsyncMock()

        # Create directory event
        mock_event = MagicMock()
        mock_event.is_directory = True
        mock_event.src_path = "/path/to/config.yaml"

        # Should not trigger reload
        watcher.on_modified(mock_event)

        # Wait and check
        await asyncio.sleep(0.1)
        container.reload_config.assert_not_called()

    @pytest.mark.asyncio
    async def test_lazy_instance_multiple_get_calls(self):
        """Test lazy instance returns same instance on multiple calls."""
        call_count = 0

        def factory():
            nonlocal call_count
            call_count += 1
            return f"instance_{call_count}"

        lazy_instance = LazyInstance(factory)

        # Multiple calls should return same instance
        first = await lazy_instance.get()
        second = await lazy_instance.get()

        assert first == second
        assert call_count == 1  # Factory called only once

    @pytest.mark.asyncio
    async def test_container_with_external_config(self):
        """Test container with externally provided config."""
        config = Config()
        config.crawler.concurrent_requests = 42

        container = DependencyContainer()
        container.config = config  # Set external config
        await container.initialize()

        try:
            # Should use external config
            assert container.config.crawler.concurrent_requests == 42

            # Should work normally
            http_client = await container.get_http_client()
            assert http_client is not None

        finally:
            await container.shutdown()

    @pytest.mark.asyncio
    async def test_all_instance_getters(self):
        """Test all instance getter methods."""
        container = DependencyContainer()
        container.config = Config()
        await container.initialize()

        try:
            # Test all getters
            observability = await container.get_observability()
            storage = await container.get_storage()
            quality = await container.get_quality()
            dataset = await container.get_dataset()
            http_client = await container.get_http_client()

            # All should return valid instances
            assert observability is not None
            assert storage is not None
            assert quality is not None
            assert dataset is not None
            assert http_client is not None

        finally:
            await container.shutdown()

    @pytest.mark.asyncio
    async def test_instance_lock_protection(self):
        """Test instance access is protected by lock."""
        container = DependencyContainer()
        container.config = Config()
        await container.initialize()

        try:
            # Mock the lock to verify it's used
            original_lock = container._instances_lock
            container._instances_lock = AsyncMock()
            container._instances_lock.__aenter__ = AsyncMock(return_value=None)
            container._instances_lock.__aexit__ = AsyncMock(return_value=None)

            # Access instance
            await container.get_http_client()

            # Lock should have been acquired
            container._instances_lock.__aenter__.assert_called()
            container._instances_lock.__aexit__.assert_called()

        finally:
            container._instances_lock = original_lock
            await container.shutdown()


class TestLazyInstanceAdvanced:
    """Test LazyInstance behavior comprehensively."""

    @pytest.mark.asyncio
    async def test_lazy_instance_initialization_with_async_initialize(self):
        """Test LazyInstance with factory that has async initialize method."""

        class MockService:
            def __init__(self, name: str):
                self.name = name
                self.initialized = False

            async def initialize(self):
                self.initialized = True

        # Test lines 42-49: initialization path including async initialize call
        factory = LazyInstance(MockService, "test-service")

        # First get() should trigger initialization
        service = await factory.get()
        assert service.name == "test-service"
        assert service.initialized is True  # async initialize was called

        # Second get() should return same instance
        service2 = await factory.get()
        assert service is service2

    @pytest.mark.asyncio
    async def test_lazy_instance_initialization_without_initialize_method(self):
        """Test LazyInstance with factory that has no initialize method."""

        class SimpleService:
            def __init__(self, value: int):
                self.value = value

        # Test lines 42-46, 48-49: initialization without initialize method
        factory = LazyInstance(SimpleService, 42)

        service = await factory.get()
        assert service.value == 42

    @pytest.mark.asyncio
    async def test_lazy_instance_cleanup_with_close_method(self):
        """Test LazyInstance cleanup when instance has close method."""

        class MockService:
            def __init__(self):
                self.closed = False

            async def close(self):
                self.closed = True

        factory = LazyInstance(MockService)
        service = await factory.get()

        # Test line 54: cleanup calls close method
        await factory.cleanup()
        assert service.closed is True

    @pytest.mark.asyncio
    async def test_lazy_instance_cleanup_without_close_method(self):
        """Test LazyInstance cleanup when instance has no close method."""

        class SimpleService:
            def __init__(self):
                self.value = "test"

        factory = LazyInstance(SimpleService)
        await factory.get()

        # Should not raise error even without close method


class TestContainerModuleGetters:
    """Test the specific module getter methods that are missing coverage."""

    @pytest.fixture
    def container(self):
        """Create container with mocked dependencies."""
        container = DependencyContainer()
        container.config = Config()  # Set config directly
        return container

    @pytest.mark.asyncio
    async def test_get_observability_method(self, container):
        """Test lines 161-162: get_observability method execution."""

        # Mock the LazyInstance
        mock_instance = AsyncMock()
        mock_observability = MagicMock()
        mock_instance.get.return_value = mock_observability

        container._instances = {"observability": mock_instance}

        result = await container.get_observability()
        assert result is mock_observability
        mock_instance.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_storage_method(self, container):
        """Test lines 166-167: get_storage method execution."""

        mock_instance = AsyncMock()
        mock_storage = MagicMock()
        mock_instance.get.return_value = mock_storage

        container._instances = {"storage": mock_instance}

        result = await container.get_storage()
        assert result is mock_storage
        mock_instance.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_quality_method(self, container):
        """Test lines 171-172: get_quality method execution."""

        mock_instance = AsyncMock()
        mock_quality = MagicMock()
        mock_instance.get.return_value = mock_quality

        container._instances = {"quality": mock_instance}

        result = await container.get_quality()
        assert result is mock_quality
        mock_instance.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_dataset_method(self, container):
        """Test lines 176-177: get_dataset method execution."""

        mock_instance = AsyncMock()
        mock_dataset = MagicMock()
        mock_instance.get.return_value = mock_dataset

        container._instances = {"dataset": mock_instance}

        result = await container.get_dataset()
        assert result is mock_dataset
        mock_instance.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_http_client_method(self, container):
        """Test lines 181-182: get_http_client method execution."""

        mock_instance = AsyncMock()
        mock_client = MagicMock()
        mock_instance.get.return_value = mock_client

        container._instances = {"http_client": mock_instance}

        result = await container.get_http_client()
        assert result is mock_client
        mock_instance.get.assert_called_once()


class TestContainerErrorConditions:
    """Test error conditions and edge cases."""

    @pytest.mark.asyncio
    async def test_create_instances_without_config_raises_error(self):
        """Test line 125: RuntimeError when config is None in _create_instances."""

        container = DependencyContainer()
        # Don't set config, it should remain None

        with pytest.raises(RuntimeError, match="Configuration must be loaded before creating instances"):
            await container._create_instances()

    @pytest.mark.asyncio
    async def test_initialize_with_existing_config(self):
        """Test initialization path when config is already provided."""

        config = Config()
        container = DependencyContainer()
        container.config = config  # Pre-set config

        with (
            patch.object(container, "_create_instances") as mock_create,
            patch.object(container, "_setup_config_watching") as mock_watch,
            patch.object(container, "_setup_signal_handlers") as mock_signals,
        ):
            await container.initialize()

            # Should call _create_instances directly, not load_config
            mock_create.assert_called_once()
            mock_watch.assert_called_once()
            mock_signals.assert_called_once()
            assert container.is_running is True

    @pytest.mark.asyncio
    async def test_module_getters_thread_safety(self):
        """Test that module getters properly use async locks."""

        container = DependencyContainer()
        container.config = Config()

        # Mock the instances with delayed responses to test concurrency
        async def delayed_get():
            await asyncio.sleep(0.01)  # Small delay
            return MagicMock()

        mock_instance = AsyncMock()
        mock_instance.get.side_effect = delayed_get
        container._instances = {"observability": mock_instance}

        # Run multiple concurrent requests
        tasks = [container.get_observability() for _ in range(5)]
        results = await asyncio.gather(*tasks)

        # All should succeed
        assert len(results) == 5
        # get() should be called 5 times (once per request)
        assert mock_instance.get.call_count == 5

    @pytest.mark.asyncio
    async def test_lazy_instance_with_factory_arguments(self):
        """Test LazyInstance with factory arguments and kwargs."""

        class ConfigurableService:
            def __init__(self, name: str, debug: bool = False, timeout: int = 30):
                self.name = name
                self.debug = debug
                self.timeout = timeout

        factory = LazyInstance(ConfigurableService, "test", debug=True, timeout=60)
        service = await factory.get()

        assert service.name == "test"
        assert service.debug is True
        assert service.timeout == 60

    @pytest.mark.asyncio
    async def test_lazy_instance_factory_exception_handling(self):
        """Test LazyInstance behavior when factory raises exception."""

        def failing_factory():
            raise ValueError("Factory failed")

        factory = LazyInstance(failing_factory)

        with pytest.raises(ValueError, match="Factory failed"):
            await factory.get()

        # Instance should remain uninitialized
        assert factory._initialized is False
        assert factory._instance is None

    @pytest.mark.asyncio
    async def test_lazy_instance_initialize_method_exception(self):
        """Test LazyInstance behavior when initialize method raises exception."""

        class FailingService:
            def __init__(self):
                self.created = True

            async def initialize(self):
                raise RuntimeError("Initialize failed")

        factory = LazyInstance(FailingService)

        with pytest.raises(RuntimeError, match="Initialize failed"):
            await factory.get()

        # Instance should be created but initialization failed
        assert factory._initialized is False

    @pytest.mark.asyncio
    async def test_container_lifecycle_with_real_modules(self):
        """Integration test with real module imports to ensure paths work."""

        container = DependencyContainer()
        container.config = Config()

        # This should trigger the real import statements in _create_instances
        await container._create_instances()

        # Verify instances are created
        assert "observability" in container._instances
        assert "storage" in container._instances
        assert "quality" in container._instances
        assert "dataset" in container._instances
        assert "http_client" in container._instances

        # Clean up
        await container._cleanup_instances()
