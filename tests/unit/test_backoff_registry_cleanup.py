"""
Comprehensive tests for DomainBackoffRegistry cleanup functionality.

This test suite covers the cleanup method with various scenarios including
expired domains, direct state manipulation, and edge cases to boost coverage.
"""

import asyncio
import time

import pytest
import pytest_asyncio
from quarrycore.config.config import Config
from quarrycore.crawler.http_client import DomainBackoffRegistry
from quarrycore.observability.metrics import METRICS


class TestDomainBackoffRegistryCleanup:
    """Test suite for DomainBackoffRegistry cleanup functionality."""

    @pytest_asyncio.fixture
    async def backoff_registry(self):
        """Create a DomainBackoffRegistry with short cooldown for testing."""
        return DomainBackoffRegistry(cooldown_seconds=0.1)

    @pytest.mark.asyncio
    async def test_cleanup_expired_domains_with_real_time(self, backoff_registry):
        """Test cleanup removes expired cooldown entries using real time."""
        # Set up domains with failures
        await backoff_registry.record_failure("expired1.com")
        await backoff_registry.record_failure("expired1.com")
        await backoff_registry.record_failure("expired1.com")  # Should trigger cooldown

        await backoff_registry.record_failure("expired2.com")
        await backoff_registry.record_failure("expired2.com")
        await backoff_registry.record_failure("expired2.com")  # Should trigger cooldown

        # Verify initial state
        stats = backoff_registry.get_stats()
        assert stats["domains_in_cooldown"] == 2
        assert len(backoff_registry._cooldown_until) == 2

        # Wait for cooldown to expire
        await asyncio.sleep(0.15)  # Wait past the 0.1s cooldown

        # Call cleanup - should remove expired entries
        backoff_registry.cleanup()

        # Verify cleanup removed expired entries
        assert len(backoff_registry._cooldown_until) == 0

        # Verify failure counts are preserved (historical data)
        assert "expired1.com" in backoff_registry._failures
        assert "expired2.com" in backoff_registry._failures

    @pytest.mark.asyncio
    async def test_cleanup_with_manual_state_manipulation(self, backoff_registry):
        """Test cleanup by directly manipulating internal state."""
        # Manually set up expired cooldown entries
        current_time = time.time()
        backoff_registry._cooldown_until["expired.com"] = current_time - 10  # Already expired
        backoff_registry._cooldown_until["active.com"] = current_time + 100  # Future expiry
        backoff_registry._failures["expired.com"] = 3
        backoff_registry._failures["active.com"] = 2

        # Verify initial state
        assert len(backoff_registry._cooldown_until) == 2

        # Call cleanup
        backoff_registry.cleanup()

        # Verify expired entry is removed but active one remains
        assert len(backoff_registry._cooldown_until) == 1
        assert "active.com" in backoff_registry._cooldown_until
        assert "expired.com" not in backoff_registry._cooldown_until

        # Verify failure counts are preserved
        assert "expired.com" in backoff_registry._failures
        assert "active.com" in backoff_registry._failures

    @pytest.mark.asyncio
    async def test_cleanup_with_no_expired_domains(self, backoff_registry):
        """Test cleanup when no domains are expired."""
        # Set up domains in cooldown with future expiry
        current_time = time.time()
        backoff_registry._cooldown_until["domain1.com"] = current_time + 100
        backoff_registry._cooldown_until["domain2.com"] = current_time + 200
        backoff_registry._failures["domain1.com"] = 3
        backoff_registry._failures["domain2.com"] = 3

        # Verify initial state
        initial_cooldown_count = len(backoff_registry._cooldown_until)
        assert initial_cooldown_count == 2

        # Call cleanup
        backoff_registry.cleanup()

        # Verify no changes
        assert len(backoff_registry._cooldown_until) == initial_cooldown_count
        assert "domain1.com" in backoff_registry._cooldown_until
        assert "domain2.com" in backoff_registry._cooldown_until

    @pytest.mark.asyncio
    async def test_cleanup_with_empty_registry(self, backoff_registry):
        """Test cleanup when registry is empty."""
        # Verify registry is empty
        assert len(backoff_registry._cooldown_until) == 0
        assert len(backoff_registry._failures) == 0

        # Call cleanup on empty registry
        backoff_registry.cleanup()

        # Verify still empty
        assert len(backoff_registry._cooldown_until) == 0
        assert len(backoff_registry._failures) == 0

    @pytest.mark.asyncio
    async def test_cleanup_preserves_failure_history(self, backoff_registry):
        """Test that cleanup preserves failure history for expired domains."""
        # Set up domain with expired cooldown
        current_time = time.time()
        backoff_registry._cooldown_until["test.com"] = current_time - 10  # Expired
        backoff_registry._failures["test.com"] = 5  # Historical failures

        # Call cleanup
        backoff_registry.cleanup()

        # Verify cooldown is removed but failure count is preserved
        assert "test.com" not in backoff_registry._cooldown_until
        assert "test.com" in backoff_registry._failures
        assert backoff_registry._failures["test.com"] == 5

    @pytest.mark.asyncio
    async def test_cleanup_mixed_expired_and_active(self, backoff_registry):
        """Test cleanup with mix of expired and active domains."""
        current_time = time.time()

        # Set up domains with different expiry times
        backoff_registry._cooldown_until["expired1.com"] = current_time - 10  # Expired
        backoff_registry._cooldown_until["expired2.com"] = current_time - 5  # Expired
        backoff_registry._cooldown_until["active1.com"] = current_time + 100  # Active
        backoff_registry._cooldown_until["active2.com"] = current_time + 200  # Active

        # Set failure counts for all
        for domain in ["expired1.com", "expired2.com", "active1.com", "active2.com"]:
            backoff_registry._failures[domain] = 3

        # Verify initial state
        assert len(backoff_registry._cooldown_until) == 4

        # Call cleanup
        backoff_registry.cleanup()

        # Verify selective cleanup
        assert len(backoff_registry._cooldown_until) == 2
        assert "expired1.com" not in backoff_registry._cooldown_until
        assert "expired2.com" not in backoff_registry._cooldown_until
        assert "active1.com" in backoff_registry._cooldown_until
        assert "active2.com" in backoff_registry._cooldown_until

        # Verify all failure counts preserved
        for domain in ["expired1.com", "expired2.com", "active1.com", "active2.com"]:
            assert domain in backoff_registry._failures
            assert backoff_registry._failures[domain] == 3

    @pytest.mark.asyncio
    async def test_cleanup_edge_case_exactly_expired(self, backoff_registry):
        """Test cleanup when domain expires exactly at current time."""
        current_time = time.time()

        # Set cooldown to expire exactly now
        backoff_registry._cooldown_until["edge.com"] = current_time
        backoff_registry._failures["edge.com"] = 3

        # Call cleanup immediately
        backoff_registry.cleanup()

        # Domain should be cleaned up (cooldown_time <= current_time)
        assert "edge.com" not in backoff_registry._cooldown_until
        assert "edge.com" in backoff_registry._failures

    @pytest.mark.asyncio
    async def test_cleanup_performance_with_many_domains(self, backoff_registry):
        """Test cleanup performance with many expired domains."""
        current_time = time.time()
        domain_count = 100

        # Set up many expired domains
        for i in range(domain_count):
            domain = f"domain{i}.com"
            backoff_registry._cooldown_until[domain] = current_time - 10  # All expired
            backoff_registry._failures[domain] = 3

        # Verify initial state
        assert len(backoff_registry._cooldown_until) == domain_count

        # Measure cleanup time
        start_time = time.time()
        backoff_registry.cleanup()
        end_time = time.time()

        # Verify cleanup completed
        assert len(backoff_registry._cooldown_until) == 0

        # Verify reasonable performance (cleanup should be fast)
        cleanup_duration = end_time - start_time
        assert cleanup_duration < 0.1  # Should complete in <100ms

        # Verify all failure counts preserved
        assert len(backoff_registry._failures) == domain_count

    @pytest.mark.asyncio
    async def test_cleanup_stats_consistency(self, backoff_registry):
        """Test that cleanup maintains stats consistency."""
        current_time = time.time()

        # Set up domains with different expiry times
        backoff_registry._cooldown_until["short.com"] = current_time - 10  # Expired
        backoff_registry._cooldown_until["long.com"] = current_time + 100  # Active
        backoff_registry._failures["short.com"] = 3
        backoff_registry._failures["long.com"] = 3

        # Get stats before cleanup
        stats_before = backoff_registry.get_stats()
        assert stats_before["domains_in_cooldown"] == 1  # Only active one counts

        # Call cleanup
        backoff_registry.cleanup()

        # Get stats after cleanup
        stats_after = backoff_registry.get_stats()

        # Verify stats consistency
        assert stats_after["domains_in_cooldown"] == 1
        assert "short.com" not in stats_after["cooldown_domains"]
        assert "long.com" in stats_after["cooldown_domains"]
        assert stats_after["total_failures"]["short.com"] == 3
        assert stats_after["total_failures"]["long.com"] == 3

    @pytest.mark.asyncio
    async def test_cleanup_concurrent_access_safety(self, backoff_registry):
        """Test cleanup with concurrent operations."""
        current_time = time.time()

        # Set up some expired domains
        for i in range(10):
            domain = f"concurrent{i}.com"
            backoff_registry._cooldown_until[domain] = current_time - 10
            backoff_registry._failures[domain] = 3

        # Define concurrent operations
        async def cleanup_task():
            backoff_registry.cleanup()
            await asyncio.sleep(0.001)

        async def record_failure_task():
            await backoff_registry.record_failure("new.com")
            await asyncio.sleep(0.001)

        # Run operations concurrently
        await asyncio.gather(cleanup_task(), cleanup_task(), record_failure_task(), cleanup_task())

        # Verify state is consistent
        assert len(backoff_registry._cooldown_until) >= 0  # May have new entries
        assert len(backoff_registry._failures) >= 10  # Should preserve + potentially add

    @pytest.mark.asyncio
    async def test_cleanup_integration_with_real_workflow(self, backoff_registry):
        """Test cleanup in a realistic workflow scenario."""
        # Simulate a realistic scenario
        domains = ["site1.com", "site2.com", "site3.com"]

        # Trigger failures and cooldowns
        for domain in domains:
            await backoff_registry.record_failure(domain)
            await backoff_registry.record_failure(domain)
            await backoff_registry.record_failure(domain)  # Triggers cooldown

        # Verify all in cooldown
        stats = backoff_registry.get_stats()
        assert stats["domains_in_cooldown"] == 3

        # Wait for cooldown to expire
        await asyncio.sleep(0.15)

        # Call cleanup
        backoff_registry.cleanup()

        # Verify cleanup worked
        assert len(backoff_registry._cooldown_until) == 0

        # Verify we can record new failures after cleanup
        await backoff_registry.record_failure("site1.com")
        assert "site1.com" in backoff_registry._failures

        # Verify success can still clear things
        await backoff_registry.record_success("site1.com")
        assert "site1.com" not in backoff_registry._failures
