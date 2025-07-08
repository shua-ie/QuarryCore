"""
Unit tests for domain backoff registry.
"""

import asyncio
import time
from unittest.mock import patch

import pytest
from quarrycore.crawler.http_client import DomainBackoffRegistry


class TestDomainBackoffRegistry:
    """Test domain backoff and cooldown behavior."""

    @pytest.fixture
    def registry(self):
        """Create backoff registry with short cooldown for testing."""
        return DomainBackoffRegistry(cooldown_seconds=1)

    @pytest.mark.asyncio
    async def test_initial_state(self, registry):
        """Test initial state of registry."""
        assert not await registry.is_in_cooldown("example.com")
        stats = registry.get_stats()
        assert stats["domains_in_cooldown"] == 0
        assert stats["total_failures"] == {}

    @pytest.mark.asyncio
    async def test_single_failure_no_cooldown(self, registry):
        """Test that single failure doesn't trigger cooldown."""
        await registry.record_failure("example.com")

        assert not await registry.is_in_cooldown("example.com")
        stats = registry.get_stats()
        assert stats["domains_in_cooldown"] == 0
        assert stats["total_failures"]["example.com"] == 1

    @pytest.mark.asyncio
    async def test_two_failures_no_cooldown(self, registry):
        """Test that two failures don't trigger cooldown."""
        await registry.record_failure("example.com")
        await registry.record_failure("example.com")

        assert not await registry.is_in_cooldown("example.com")
        stats = registry.get_stats()
        assert stats["domains_in_cooldown"] == 0
        assert stats["total_failures"]["example.com"] == 2

    @pytest.mark.asyncio
    async def test_three_failures_trigger_cooldown(self, registry):
        """Test that three failures trigger cooldown."""
        await registry.record_failure("example.com")
        await registry.record_failure("example.com")
        await registry.record_failure("example.com")

        assert await registry.is_in_cooldown("example.com")
        stats = registry.get_stats()
        assert stats["domains_in_cooldown"] == 1
        assert stats["total_failures"]["example.com"] == 3

    @pytest.mark.asyncio
    async def test_success_clears_failures(self, registry):
        """Test that success clears failure count."""
        await registry.record_failure("example.com")
        await registry.record_failure("example.com")
        await registry.record_success("example.com")

        assert not await registry.is_in_cooldown("example.com")
        stats = registry.get_stats()
        assert stats["domains_in_cooldown"] == 0
        assert "example.com" not in stats["total_failures"]

    @pytest.mark.asyncio
    async def test_cooldown_expires(self, registry):
        """Test that cooldown expires after timeout."""
        # Trigger cooldown
        await registry.record_failure("example.com")
        await registry.record_failure("example.com")
        await registry.record_failure("example.com")

        assert await registry.is_in_cooldown("example.com")

        # Wait for cooldown to expire
        await asyncio.sleep(1.1)

        assert not await registry.is_in_cooldown("example.com")

    @pytest.mark.asyncio
    async def test_multiple_domains_independent(self, registry):
        """Test that domains are tracked independently."""
        # Trigger cooldown for example.com
        await registry.record_failure("example.com")
        await registry.record_failure("example.com")
        await registry.record_failure("example.com")

        # Add one failure for test.com
        await registry.record_failure("test.com")

        assert await registry.is_in_cooldown("example.com")
        assert not await registry.is_in_cooldown("test.com")

        stats = registry.get_stats()
        assert stats["domains_in_cooldown"] == 1
        assert stats["total_failures"]["example.com"] == 3
        assert stats["total_failures"]["test.com"] == 1

    @pytest.mark.asyncio
    async def test_success_during_cooldown_clears_state(self, registry):
        """Test that success during cooldown clears all state."""
        # Trigger cooldown
        await registry.record_failure("example.com")
        await registry.record_failure("example.com")
        await registry.record_failure("example.com")

        assert await registry.is_in_cooldown("example.com")

        # Record success
        await registry.record_success("example.com")

        assert not await registry.is_in_cooldown("example.com")
        stats = registry.get_stats()
        assert stats["domains_in_cooldown"] == 0
        assert "example.com" not in stats["total_failures"]

    @pytest.mark.asyncio
    async def test_concurrent_access(self, registry):
        """Test concurrent access to registry."""

        async def record_failures():
            for _ in range(10):
                await registry.record_failure("example.com")

        async def record_successes():
            for _ in range(5):
                await registry.record_success("example.com")
                await asyncio.sleep(0.01)

        # Run concurrent operations
        await asyncio.gather(record_failures(), record_successes())

        # Registry should be in a consistent state
        stats = registry.get_stats()
        assert isinstance(stats["domains_in_cooldown"], int)
        assert isinstance(stats["total_failures"], dict)

    @pytest.mark.asyncio
    async def test_cooldown_time_accuracy(self, registry):
        """Test that cooldown time is approximately correct."""
        # Use longer cooldown for accuracy test
        long_registry = DomainBackoffRegistry(cooldown_seconds=0.5)

        # Trigger cooldown
        await long_registry.record_failure("example.com")
        await long_registry.record_failure("example.com")
        await long_registry.record_failure("example.com")

        start_time = time.time()
        assert await long_registry.is_in_cooldown("example.com")

        # Wait for cooldown to expire
        await asyncio.sleep(0.6)

        assert not await long_registry.is_in_cooldown("example.com")

        # Should be approximately 0.5 seconds
        elapsed = time.time() - start_time
        assert 0.5 < elapsed < 1.0

    @pytest.mark.asyncio
    async def test_stats_accuracy(self, registry):
        """Test that statistics are accurate."""
        # Create mixed state
        await registry.record_failure("example.com")
        await registry.record_failure("example.com")
        await registry.record_failure("example.com")  # In cooldown

        await registry.record_failure("test.com")
        await registry.record_failure("test.com")  # Not in cooldown

        await registry.record_success("other.com")  # No failures

        stats = registry.get_stats()

        assert stats["domains_in_cooldown"] == 1
        assert stats["total_failures"]["example.com"] == 3
        assert stats["total_failures"]["test.com"] == 2
        assert "other.com" not in stats["total_failures"]

        # Should have cooldown info for example.com
        assert "example.com" in stats["cooldown_domains"]
        assert stats["cooldown_domains"]["example.com"] > 0

    @pytest.mark.asyncio
    async def test_expired_cooldown_cleanup(self, registry):
        """Test that expired cooldowns are cleaned up."""
        # Use very short cooldown
        short_registry = DomainBackoffRegistry(cooldown_seconds=0.1)

        # Trigger cooldown
        await short_registry.record_failure("example.com")
        await short_registry.record_failure("example.com")
        await short_registry.record_failure("example.com")

        # Wait for expiration
        await asyncio.sleep(0.2)

        # First call should clean up expired cooldown
        assert not await short_registry.is_in_cooldown("example.com")

        # Stats should reflect cleanup
        stats = short_registry.get_stats()
        assert stats["domains_in_cooldown"] == 0
        assert "example.com" not in stats["cooldown_domains"]

    @pytest.mark.asyncio
    async def test_default_cooldown_duration(self):
        """Test default cooldown duration."""
        default_registry = DomainBackoffRegistry()

        # Should have default 120 seconds
        assert default_registry.cooldown_seconds == 120

    @pytest.mark.asyncio
    async def test_zero_cooldown_duration(self):
        """Test zero cooldown duration."""
        zero_registry = DomainBackoffRegistry(cooldown_seconds=0)

        # Trigger failures
        await zero_registry.record_failure("example.com")
        await zero_registry.record_failure("example.com")
        await zero_registry.record_failure("example.com")

        # Should immediately not be in cooldown
        assert not await zero_registry.is_in_cooldown("example.com")
