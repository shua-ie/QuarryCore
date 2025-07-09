"""
Production-grade dependency injection container for QuarryCore modules.
"""

from __future__ import annotations

import asyncio
import signal
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable, Dict, Generic, List, Optional, TypeVar
from uuid import uuid4

import structlog
from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from quarrycore.config import Config

if TYPE_CHECKING:
    from quarrycore.crawler.http_client import HttpClient
    from quarrycore.dataset import DatasetConstructor
    from quarrycore.observability import ObservabilityManager
    from quarrycore.quality import QualityAssessor
    from quarrycore.recovery.dead_letter_service import SQLiteDeadLetterService
    from quarrycore.storage import StorageManager

T = TypeVar("T")


class LazyInstance(Generic[T]):
    """Lazy-loaded instance with lifecycle management."""

    def __init__(self, factory: Callable[..., T], *args: Any, **kwargs: Any) -> None:
        self._factory = factory
        self._args = args
        self._kwargs = kwargs
        self._instance: Optional[T] = None
        self._initialized = False

    async def get(self) -> T:
        """Get or create the instance."""
        if not self._initialized:
            self._instance = self._factory(*self._args, **self._kwargs)
            if hasattr(self._instance, "initialize") and callable(getattr(self._instance, "initialize", None)):
                await self._instance.initialize()  # type: ignore
            self._initialized = True
        # At this point _instance cannot be None due to the check above
        assert self._instance is not None
        return self._instance

    async def cleanup(self) -> None:
        """Clean up the instance."""
        if self._instance and hasattr(self._instance, "close") and callable(getattr(self._instance, "close", None)):
            await self._instance.close()  # type: ignore
        self._instance = None
        self._initialized = False


class ConfigWatcher(FileSystemEventHandler):
    """Watches configuration file for changes."""

    def __init__(self, container: DependencyContainer) -> None:
        self.container = container
        self.logger = structlog.get_logger(self.__class__.__name__)

    def on_modified(self, event: FileSystemEvent) -> None:
        if not event.is_directory and str(event.src_path).endswith((".yaml", ".yml")):
            self.logger.info("Configuration file changed, reloading", path=event.src_path)
            asyncio.create_task(self.container.reload_config())


class DependencyContainer:
    """
    Central dependency injection container managing all QuarryCore modules.
    Provides lazy initialization, lifecycle management, and configuration hot-reloading.
    """

    def __init__(self, config_path: Optional[Path] = None) -> None:
        self.config_path = config_path
        self.config: Optional[Config] = None
        self.logger = structlog.get_logger(self.__class__.__name__)

        # Lazy-loaded module instances
        self._instances: Dict[str, LazyInstance[Any]] = {}
        self._instances_lock = asyncio.Lock()
        self._observer: Optional[Any] = None  # Observer type
        self._shutdown_handlers: List[Callable[[], Any]] = []

        # Pipeline state
        self.pipeline_id = str(uuid4())
        self.is_running = False

    async def initialize(self) -> None:
        """Initialize the container and load configuration."""
        # Only load config if not already provided
        if self.config is None:
            await self.load_config()
        else:
            # Config already provided, just create instances
            await self._create_instances()

        await self._setup_config_watching()
        await self._setup_signal_handlers()

        self.is_running = True

        self.logger.info(
            "Dependency container initialized",
            pipeline_id=self.pipeline_id,
            config_path=str(self.config_path) if self.config_path else "default",
        )

    async def load_config(self) -> None:
        """Load or reload configuration."""
        if self.config_path and self.config_path.exists():
            self.config = Config.from_yaml(self.config_path)
        else:
            self.config = Config()

        await self._create_instances()

    async def _create_instances(self) -> None:
        """Create lazy instances with current configuration."""
        if self.config is None:
            raise RuntimeError("Configuration must be loaded before creating instances")

        # Clear existing instances on config reload
        await self._cleanup_instances()
        self._instances.clear()

        # Import modules only when needed to avoid circular imports
        from quarrycore.crawler.http_client import HttpClient
        from quarrycore.dataset import DatasetConstructor
        from quarrycore.observability import ObservabilityManager
        from quarrycore.quality import QualityAssessor
        from quarrycore.recovery.dead_letter_service import SQLiteDeadLetterService
        from quarrycore.storage import StorageManager

        # Create lazy instances with new configuration
        self._instances = {
            "observability": LazyInstance(ObservabilityManager, self.config.monitoring),
            "storage": LazyInstance(StorageManager, self.config.storage),
            "quality": LazyInstance(QualityAssessor, self.config.quality),
            "dataset": LazyInstance(DatasetConstructor, self.config.dataset),
            "http_client": LazyInstance(HttpClient, self.config),
            "dead_letter": LazyInstance(
                SQLiteDeadLetterService, self.config.storage.hot.db_path.parent / "dead_letter.db"
            ),
            # Add other modules as they become available
        }

    async def reload_config(self) -> None:
        """Hot-reload configuration and reinitialize affected modules."""
        old_config = self.config
        await self.load_config()

        self.logger.info(
            "Configuration reloaded",
            pipeline_id=self.pipeline_id,
            changes_detected=old_config != self.config,
        )

    async def get_observability(self) -> ObservabilityManager:
        """Get the observability manager instance."""
        async with self._instances_lock:
            return await self._instances["observability"].get()  # type: ignore

    async def get_storage(self) -> StorageManager:
        """Get the storage manager instance."""
        async with self._instances_lock:
            return await self._instances["storage"].get()  # type: ignore

    async def get_quality(self) -> QualityAssessor:
        """Get the quality assessor instance."""
        async with self._instances_lock:
            return await self._instances["quality"].get()  # type: ignore

    async def get_dataset(self) -> DatasetConstructor:
        """Get the dataset constructor instance."""
        async with self._instances_lock:
            return await self._instances["dataset"].get()  # type: ignore

    async def get_http_client(self) -> HttpClient:
        """Get the HTTP client instance."""
        async with self._instances_lock:
            return await self._instances["http_client"].get()  # type: ignore

    async def get_dead_letter(self) -> SQLiteDeadLetterService:
        """Get the dead letter service instance."""
        async with self._instances_lock:
            return await self._instances["dead_letter"].get()  # type: ignore

    @asynccontextmanager
    async def lifecycle(self) -> AsyncIterator[DependencyContainer]:
        """Context manager for proper lifecycle management."""
        try:
            await self.initialize()
            self.is_running = True
            yield self
        finally:
            await self.shutdown()

    async def shutdown(self) -> None:
        """Graceful shutdown of all managed instances."""
        if not self.is_running:
            return

        self.logger.info("Shutting down dependency container", pipeline_id=self.pipeline_id)

        # Stop config watching
        if self._observer:
            self._observer.stop()
            self._observer.join()

        # Run shutdown handlers first
        for handler in self._shutdown_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler()
                else:
                    handler()
            except Exception as e:
                self.logger.error("Error in shutdown handler", error=str(e))

        # Cleanup all instances
        await self._cleanup_instances()

        self.is_running = False
        self.logger.info("Dependency container shutdown complete")

    async def _setup_config_watching(self) -> None:
        """Set up file system watching for configuration changes."""
        if not self.config_path:
            return

        self._observer = Observer()
        handler = ConfigWatcher(self)
        self._observer.schedule(handler, str(self.config_path.parent), recursive=False)
        self._observer.start()

    async def _setup_signal_handlers(self) -> None:
        """Set up signal handlers for graceful shutdown."""

        def signal_handler(signum: int, frame: Any) -> None:
            self.logger.info(f"Received signal {signum}, initiating shutdown")
            asyncio.create_task(self.shutdown())

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

    async def _cleanup_instances(self) -> None:
        """Clean up all managed instances."""
        for name, instance in self._instances.items():
            try:
                await instance.cleanup()
            except Exception as e:
                self.logger.error(f"Error cleaning up {name}", error=str(e))

    def add_shutdown_handler(self, handler: Callable[[], Any]) -> None:
        """Add a custom shutdown handler."""
        self._shutdown_handlers.append(handler)

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of all managed components."""
        return {
            "pipeline_id": self.pipeline_id,
            "is_running": self.is_running,
            "config_loaded": self.config is not None,
            "instances_count": len(self._instances),
            "config_path": str(self.config_path) if self.config_path else None,
        }
