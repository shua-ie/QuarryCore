"""
Comprehensive tests for HTTP client branch coverage gaps.

This test suite specifically targets the missing branches identified in the Phase-2 audit
to raise branch coverage from 79.7% to ≥85%.
"""

import asyncio
import time
from unittest.mock import AsyncMock, patch

import aiohttp
import pytest
import pytest_asyncio
from aioresponses import CallbackResult, aioresponses
from quarrycore.config.config import Config
from quarrycore.crawler.http_client import DomainBackoffRegistry, HttpClient
from quarrycore.observability.metrics import METRICS


class TestHttpClientBranchCoverage:
    """Test suite targeting specific uncovered branches in HttpClient."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        config = Config()
        config.debug.test_mode = True
        config.crawler.max_retries = 2
        config.crawler.timeout = 5.0
        config.crawler.respect_robots = True
        return config

    @pytest.fixture
    async def http_client(self, config):
        """Create and initialize HTTP client."""
        client = HttpClient(config)
        await client.initialize()
        yield client
        await client.close()

    @pytest.mark.asyncio
    async def test_malformed_url_edge_cases(self, http_client):
        """Test malformed URL handling to cover parsing edge cases."""
        # Test completely invalid URLs
        malformed_urls = [
            "not-a-url-at-all",
            "://missing-scheme.com",
            "http://",
            "https://",
            "",
            "   ",
            "ftp://unsupported-scheme.com",
            "http:///no-domain",
            "https://[invalid-ipv6",
            "https://[::1::invalid]:8080",
        ]

        for url in malformed_urls:
            response = await http_client.fetch(url)
            assert response.status == 0
            assert response.body == b""
            # Malformed URLs may go through retry logic for some schemes like 'ftp://'
            assert response.attempts >= 1

    @pytest.mark.asyncio
    async def test_proxy_scheme_mismatch(self, config):
        """Test proxy scheme mismatch scenarios."""
        # Set up HTTP proxy but try HTTPS URL
        with patch.dict("os.environ", {"QUARRY_HTTP_PROXIES": "http://proxy.example.com:8080"}):
            async with HttpClient(config) as client:
                # This should trigger proxy scheme mismatch logic
                with aioresponses() as m:
                    m.get("https://example.com/test", status=200, body="Success")

                    response = await client.fetch("https://example.com/test")
                    assert response.status == 200

    @pytest.mark.asyncio
    async def test_proxy_deterministic_selection(self, config):
        """Test deterministic proxy selection in test mode."""
        with patch.dict(
            "os.environ",
            {"QUARRY_HTTP_PROXIES": "http://proxy1.com:8080,https://proxy2.com:8080,http://proxy3.com:8080"},
        ):
            async with HttpClient(config) as client:
                # Test deterministic selection for different schemes
                http_proxy = client._select_proxy_deterministic("http://example.com")
                https_proxy = client._select_proxy_deterministic("https://example.com")

                assert http_proxy == "http://proxy1.com:8080"  # First HTTP proxy
                assert https_proxy == "https://proxy2.com:8080"  # First HTTPS proxy

    @pytest.mark.asyncio
    async def test_proxy_no_scheme_match_fallback(self, config):
        """Test proxy fallback when no scheme matches."""
        with patch.dict("os.environ", {"QUARRY_HTTP_PROXIES": "socks5://proxy.example.com:1080"}):
            async with HttpClient(config) as client:
                # Should fall back to first proxy even with scheme mismatch
                proxy = client._select_proxy_deterministic("https://example.com")
                assert proxy == "socks5://proxy.example.com:1080"

    @pytest.mark.asyncio
    async def test_retry_exhaustion_with_metrics(self, http_client):
        """Test retry exhaustion path with metrics tracking."""
        with aioresponses() as m:
            # Set up responses that always fail with 503 - ensure proper URL matching
            url = "https://example.com/fail"
            for _ in range(10):  # More than enough for retries
                m.get(url, status=503, body="Service Unavailable")

            response = await http_client.fetch(url)

            # Should exhaust retries and return final HTTP status (503) or error status (0)
            assert response.status in [0, 503], f"Expected status 0 or 503, got {response.status}"
            assert response.attempts >= 2  # Should have attempted multiple times

    @pytest.mark.asyncio
    async def test_timeout_in_retry_loop(self, http_client):
        """Test timeout handling in retry scenarios."""
        call_count = 0

        async def timeout_callback(url, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                # First two calls timeout
                await asyncio.sleep(10)  # Longer than timeout
                return CallbackResult(status=200, body="Too slow")
            else:
                # Third call succeeds quickly
                return CallbackResult(status=200, body="Success")

        with aioresponses() as m:
            m.get("https://example.com/timeout", callback=timeout_callback)

            response = await http_client.fetch("https://example.com/timeout", timeout=0.1)

            # Should eventually succeed or fail after retries
            assert response.attempts > 1

    @pytest.mark.asyncio
    async def test_metrics_error_conditions(self, config):
        """Test error handling when metrics system fails."""
        async with HttpClient(config) as client:
            # Simulate metrics errors by temporarily removing metrics
            original_metrics = METRICS.copy()
            METRICS.clear()

            try:
                with aioresponses() as m:
                    m.get("https://example.com/test", status=200, body="Success")

                    # Should handle missing metrics gracefully
                    response = await client.fetch("https://example.com/test")
                    assert response.status == 200
            finally:
                # Restore metrics
                METRICS.update(original_metrics)

    @pytest.mark.asyncio
    async def test_robots_cache_initialization_edge_case(self, config):
        """Test robots cache initialization when session is None."""
        client = HttpClient(config)

        # Before initialization, robots_cache should return None
        assert client.robots_cache is None

        await client.initialize()

        # After initialization, should have robots cache
        assert client.robots_cache is not None

        await client.close()

    @pytest.mark.asyncio
    async def test_connector_close_with_mock_protection(self, config):
        """Test connector close protection for mocked connectors."""
        client = HttpClient(config)

        # Mock the connector to simulate test environment
        client.connector._mock_name = "mocked_connector"

        await client.initialize()
        await client.close()

        # Should not raise error even with mocked connector

    @pytest.mark.asyncio
    async def test_session_initialization_failure(self, config):
        """Test handling of session initialization edge cases."""
        client = HttpClient(config)

        # Test multiple initialization calls
        await client.initialize()
        await client.initialize()  # Should not reinitialize

        assert client.session is not None
        assert client._is_initialized is True

        await client.close()

    @pytest.mark.asyncio
    async def test_domain_semaphore_creation_race_condition(self, http_client):
        """Test concurrent domain semaphore creation."""
        domain = "example.com"

        # Create multiple concurrent semaphore requests
        tasks = [http_client._get_domain_semaphore(domain) for _ in range(10)]

        semaphores = await asyncio.gather(*tasks)

        # All should be the same semaphore instance
        first_semaphore = semaphores[0]
        for semaphore in semaphores[1:]:
            assert semaphore is first_semaphore

    @pytest.mark.asyncio
    async def test_robots_txt_cache_edge_cases(self, http_client):
        """Test robots.txt cache error handling."""
        with aioresponses() as m:
            # Mock robots.txt that returns various error conditions
            m.get("https://example.com/robots.txt", status=404)  # No robots.txt
            m.get("https://example.com/blocked", status=200, body="Content")

            # Should allow access when robots.txt is missing
            response = await http_client.fetch("https://example.com/blocked")
            assert response.status == 200

    @pytest.mark.asyncio
    async def test_meta_robots_noindex_detection(self, http_client):
        """Test meta robots noindex detection branch."""
        html_with_noindex = """
        <html>
        <head>
            <meta name="robots" content="noindex, nofollow">
        </head>
        <body>Content</body>
        </html>
        """

        with aioresponses() as m:
            m.get("https://example.com/robots.txt", status=404)
            m.get("https://example.com/noindex", status=200, body=html_with_noindex)

            response = await http_client.fetch("https://example.com/noindex")
            # Should be blocked by meta robots
            assert response.status == 999

    @pytest.mark.asyncio
    async def test_meta_robots_parsing_error(self, http_client):
        """Test meta robots parsing error handling."""
        invalid_html = b"<invalid><html><meta name="

        with aioresponses() as m:
            m.get("https://example.com/robots.txt", status=404)
            m.get("https://example.com/invalid", status=200, body=invalid_html)

            response = await http_client.fetch("https://example.com/invalid")
            # Should succeed despite parsing error
            assert response.status == 200

    @pytest.mark.asyncio
    async def test_domain_inflight_tracking_cleanup(self, http_client):
        """Test domain in-flight request tracking and cleanup."""
        with aioresponses() as m:
            m.get("https://example.com/test", status=200, body="Success")

            # Check initial state
            stats = http_client.get_stats()
            assert "example.com" not in stats["domain_inflight"]

            response = await http_client.fetch("https://example.com/test")
            assert response.status == 200

            # Should be cleaned up after request
            stats = http_client.get_stats()
            assert "example.com" not in stats["domain_inflight"]

    @pytest.mark.asyncio
    async def test_backoff_record_success_cleanup(self, http_client):
        """Test backoff registry success cleanup."""
        domain = "example.com"
        registry = http_client.backoff_registry

        # Record some failures first
        await registry.record_failure(domain)
        await registry.record_failure(domain)

        stats = registry.get_stats()
        assert domain in stats["total_failures"]

        # Record success should clean up
        await registry.record_success(domain)

        stats = registry.get_stats()
        assert domain not in stats["total_failures"]

    @pytest.mark.asyncio
    async def test_request_without_initialization(self, config):
        """Test error handling for uninitialized client."""
        client = HttpClient(config)

        with pytest.raises(RuntimeError, match="HTTP client not initialized"):
            await client.fetch("https://example.com/test")

    @pytest.mark.asyncio
    async def test_perform_request_without_initialization(self, config):
        """Test _perform_request error handling for uninitialized client."""
        client = HttpClient(config)

        with pytest.raises(RuntimeError, match="HTTP client not initialized"):
            await client._perform_request("https://example.com/test", 30.0)

    @pytest.mark.asyncio
    async def test_concurrent_exception_handling(self, http_client):
        """Test exception handling with concurrent requests."""

        async def error_callback(url, **kwargs):
            raise aiohttp.ClientError("Simulated network error")

        with aioresponses() as m:
            m.get("https://example.com/error", callback=error_callback)

            # Should handle the exception gracefully
            response = await http_client.fetch("https://example.com/error")
            assert response.status == 0

    @pytest.mark.asyncio
    async def test_unicode_content_handling(self, http_client):
        """Test handling of Unicode content in responses."""
        unicode_content = "测试内容 🚀".encode("utf-8")

        with aioresponses() as m:
            m.get("https://example.com/unicode", status=200, body=unicode_content)

            response = await http_client.fetch("https://example.com/unicode")
            assert response.status == 200
            assert response.body == unicode_content

    @pytest.mark.asyncio
    async def test_backoff_cooldown_expiration(self):
        """Test backoff registry cooldown expiration logic."""
        registry = DomainBackoffRegistry(cooldown_seconds=0.1)  # Short cooldown
        domain = "example.com"

        # Trigger cooldown
        for _ in range(3):
            await registry.record_failure(domain)

        assert await registry.is_in_cooldown(domain)

        # Wait for cooldown to expire
        await asyncio.sleep(0.2)

        # Should no longer be in cooldown
        assert not await registry.is_in_cooldown(domain)

    @pytest.mark.asyncio
    async def test_cleanup_expired_cooldowns(self):
        """Test DomainBackoffRegistry cleanup method."""
        registry = DomainBackoffRegistry(cooldown_seconds=0.1)
        domain = "example.com"

        # Trigger cooldown
        for _ in range(3):
            await registry.record_failure(domain)

        # Wait for expiration
        await asyncio.sleep(0.2)

        # Should have expired entry
        # Cleanup should remove expired entries
        registry.cleanup()

        stats_after = registry.get_stats()
        assert stats_after["domains_in_cooldown"] == 0
