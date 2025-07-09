"""
Integration test for crawler pipeline end-to-end functionality.
"""

import asyncio
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from quarrycore.config.config import Config
from quarrycore.container import DependencyContainer
from quarrycore.observability.metrics import METRICS, _create_metrics
from quarrycore.pipeline import Pipeline


class TestCrawlerPipelineE2E:
    """Test end-to-end pipeline with HTTP client integration."""

    @pytest.fixture(autouse=True)
    def setup_metrics(self):
        """Ensure metrics are available for testing."""
        # Temporarily disable test mode to get metrics
        original_test_mode = os.environ.get("QUARRY_TEST_MODE", "0")
        os.environ["QUARRY_TEST_MODE"] = "0"

        # Initialize metrics if empty
        if not METRICS:
            METRICS.update(_create_metrics())

        yield

        # Restore original test mode
        os.environ["QUARRY_TEST_MODE"] = original_test_mode

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def config(self, temp_dir):
        """Create test configuration."""
        config = Config()
        config.crawler.respect_robots = False  # Disable robots for testing
        config.crawler.max_concurrency_per_domain = 2
        config.crawler.max_retries = 2
        config.crawler.backoff_cooldown_seconds = 1
        config.crawler.user_agent = "TestBot/1.0"

        # Configure storage paths
        config.storage.hot.db_path = temp_dir / "test.db"
        config.storage.warm.base_path = temp_dir / "parquet"
        config.storage.backup.path = str(temp_dir / "backups")

        # Configure monitoring
        config.monitoring.log_file = str(temp_dir / "test.log")
        config.monitoring.prometheus_port = None  # Disable prometheus for testing

        return config

    @pytest.fixture
    async def container(self, config):
        """Create dependency container."""
        container = DependencyContainer()
        container.config = config
        await container.initialize()
        yield container
        await container.shutdown()

    @pytest.fixture
    async def pipeline(self, container):
        """Create pipeline instance."""
        pipeline = Pipeline(container)
        yield pipeline

    @pytest.mark.asyncio
    async def test_pipeline_uses_http_client(self, pipeline):
        """Test pipeline uses HTTP client instead of basic httpx."""
        # Mock HTTP client
        mock_client = AsyncMock()
        mock_client.fetch = AsyncMock(
            return_value=MagicMock(
                status=200,
                body=b"<html><body>Test</body></html>",
                headers={"Content-Type": "text/html"},
                final_url="https://example.com/page1",
            )
        )

        with patch.object(pipeline.container, "get_http_client", return_value=mock_client):
            result = await pipeline.run(["https://example.com/page1"], batch_size=1)

            # Verify HTTP client was used
            assert mock_client.fetch.called
            assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_domain_concurrency_limits(self, pipeline):
        """Test that domain concurrency limits are respected."""
        # Get HTTP client to verify configuration
        http_client = await pipeline.container.get_http_client()

        # Verify configuration
        assert http_client.crawler_config.max_concurrency_per_domain == 2
        assert http_client.crawler_config.max_retries == 2

        # Test domain semaphore creation
        semaphore = await http_client._get_domain_semaphore("example.com")
        assert semaphore._value == 2  # max_concurrency_per_domain

    @pytest.mark.asyncio
    async def test_backoff_registry_behavior(self, pipeline):
        """Test backoff registry behavior."""
        http_client = await pipeline.container.get_http_client()
        registry = http_client.backoff_registry

        # Initial state
        assert not await registry.is_in_cooldown("example.com")

        # Record failures
        await registry.record_failure("example.com")
        await registry.record_failure("example.com")
        await registry.record_failure("example.com")

        # Should be in cooldown
        assert await registry.is_in_cooldown("example.com")

        # Success should clear cooldown
        await registry.record_success("example.com")
        assert not await registry.is_in_cooldown("example.com")

    @pytest.mark.asyncio
    async def test_metrics_integration(self, pipeline):
        """Test that metrics are available."""
        # Check that crawler metrics are defined
        expected_metrics = [
            "crawler_fetch_latency_seconds",
            "crawler_responses_total",
            "crawler_in_flight_requests",
            "crawler_domain_backoff_total",
        ]

        for metric_name in expected_metrics:
            assert metric_name in METRICS, f"Missing metric: {metric_name}"

    @pytest.mark.asyncio
    async def test_http_client_configuration(self, pipeline):
        """Test HTTP client configuration."""
        http_client = await pipeline.container.get_http_client()

        # Verify configuration
        assert http_client.crawler_config.user_agent == "TestBot/1.0"
        assert http_client.crawler_config.timeout == 30.0
        assert http_client.crawler_config.max_retries == 2
        assert http_client.crawler_config.backoff_cooldown_seconds == 1
        assert not http_client.crawler_config.respect_robots

    @pytest.mark.asyncio
    async def test_http_client_lifecycle(self, pipeline):
        """Test HTTP client lifecycle management."""
        http_client = await pipeline.container.get_http_client()

        # Should be initialized
        assert http_client.session is not None

        # Should have proper connector
        assert http_client.connector is not None

        # Should have backoff registry
        assert http_client.backoff_registry is not None

    @pytest.mark.asyncio
    async def test_proxy_configuration(self, pipeline):
        """Test proxy configuration."""
        http_client = await pipeline.container.get_http_client()

        # By default, no proxies
        assert len(http_client.proxies) == 0

        # Test proxy parsing would work
        with patch.dict("os.environ", {"QUARRY_HTTP_PROXIES": "http://proxy1:8080,https://proxy2:8080"}):
            new_client = type(http_client)(pipeline.container.config)
            assert len(new_client.proxies) == 2

    @pytest.mark.asyncio
    async def test_stats_collection(self, pipeline):
        """Test that stats are collected properly."""
        http_client = await pipeline.container.get_http_client()

        # Get initial stats
        stats = http_client.get_stats()

        # Should have expected structure
        assert "in_flight_requests" in stats
        assert "domain_semaphores" in stats
        assert "proxy_count" in stats
        assert "backoff_stats" in stats

        # Should be valid values
        assert isinstance(stats["in_flight_requests"], int)
        assert isinstance(stats["domain_semaphores"], int)
        assert isinstance(stats["proxy_count"], int)
        assert isinstance(stats["backoff_stats"], dict)

    @pytest.mark.asyncio
    async def test_pipeline_protocol_compliance(self, pipeline):
        """Test that pipeline maintains protocol compliance."""
        # Mock successful HTTP client
        mock_client = AsyncMock()
        mock_client.fetch = AsyncMock(
            return_value=MagicMock(
                status=200,
                body=b"<html><head><title>Test</title></head><body>Content</body></html>",
                headers={"Content-Type": "text/html"},
                final_url="https://example.com/page1",
            )
        )

        with patch.object(pipeline.container, "get_http_client", return_value=mock_client):
            result = await pipeline.run(["https://example.com/page1"], batch_size=1)

            # Verify protocol compliance
            expected_keys = ["job_id", "pipeline_id", "status", "processed_count", "failed_count", "duration"]
            for key in expected_keys:
                assert key in result

            # Verify types
            assert isinstance(result["processed_count"], int)
            assert isinstance(result["failed_count"], int)
            assert isinstance(result["duration"], (int, float))
            assert result["status"] in ["completed", "failed"]
