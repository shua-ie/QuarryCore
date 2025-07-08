"""
Advanced property-based and edge case tests for HTTP client.

These tests use hypothesis for property-based testing and cover edge cases
to achieve 90%+ branch coverage on the HTTP client.
"""

import asyncio
import os
import random
import time
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from aioresponses import CallbackResult, aioresponses
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.strategies import composite
from quarrycore.config.config import Config
from quarrycore.crawler.http_client import DomainBackoffRegistry, HttpClient

from tests.helpers import histogram_observes, metric_delta, metric_increases


@pytest.mark.unit
class TestHttpClientAdvanced:
    """Advanced behavioral tests with property-based testing."""

    @pytest.mark.asyncio
    async def test_unsupported_url_scheme(self, http_client):
        """Test handling of unsupported URL schemes."""
        response = await http_client.fetch("ftp://example.com/file.txt")

        # Should handle gracefully with error status
        assert response.status == 0
        assert response.attempts > 0
        assert response.body == b""

    @pytest.mark.asyncio
    async def test_malformed_url_handling(self, http_client):
        """Test various malformed URL patterns."""
        malformed_urls = [
            "not-a-url",
            "://missing-scheme",
            "http://",
            "http:///no-host",
            "http://[invalid-ipv6",
            "",
            " ",
        ]

        for url in malformed_urls:
            response = await http_client.fetch(url)
            # Should handle gracefully with error status
            assert response.status in (0, 999)  # Different error statuses for different malformed URLs
            assert response.body == b""

    @pytest.mark.asyncio
    async def test_robots_txt_timeout_fallback(self, http_client):
        """Test robots.txt timeout handling."""
        # Enable robots checking
        http_client.crawler_config.respect_robots = True

        call_count = 0

        def slow_robots_callback(url, **kwargs):
            nonlocal call_count
            call_count += 1
            # Simulate timeout on robots.txt by delaying
            import time

            time.sleep(15)  # Longer than timeout
            return CallbackResult(status=200, body="User-agent: *\nAllow: /")

        with aioresponses() as m:
            m.get("https://example.com/robots.txt", callback=slow_robots_callback)
            m.get("https://example.com/page", status=200, body="Success")

            response = await http_client.fetch("https://example.com/page")

            # Should allow by default when robots.txt times out
            assert response.status == 200
            assert response.body == b"Success"

    @pytest.mark.asyncio
    async def test_4xx_error_without_body(self, http_client):
        """Test 4xx errors without response body."""
        with aioresponses() as m:
            m.get("https://example.com/page", status=404, body="")

            response = await http_client.fetch("https://example.com/page")

            assert response.status == 404
            assert response.body == b""
            assert response.attempts == 1  # No retry for 4xx

    @pytest.mark.asyncio
    async def test_concurrent_domain_requests(self, http_client):
        """Test concurrent requests to same domain are limited."""
        with aioresponses() as m:
            # Add multiple responses for the same domain
            for i in range(5):
                m.get(f"https://example.com/page{i}", status=200, body=f"Response {i}")

            # Fire concurrent requests
            tasks = [http_client.fetch(f"https://example.com/page{i}") for i in range(5)]

            start_time = time.time()
            responses = await asyncio.gather(*tasks)
            end_time = time.time()

            # All should succeed
            assert all(r.status == 200 for r in responses)

            # Should have completed in reasonable time
            assert end_time - start_time < 10.0  # Reasonable completion time

    @pytest.mark.asyncio
    async def test_metrics_validation(self, http_client):
        """Test metrics are properly recorded."""
        from quarrycore.observability.metrics import METRICS

        if "crawler_responses_total" in METRICS:
            with metric_delta(METRICS["crawler_responses_total"].labels(status_class="2xx")):
                with aioresponses() as m:
                    m.get("https://example.com/page", status=200, body="Success")
                    await http_client.fetch("https://example.com/page")

        if "crawler_fetch_latency_seconds" in METRICS:
            with histogram_observes(METRICS["crawler_fetch_latency_seconds"]):
                with aioresponses() as m:
                    m.get("https://example.com/page", status=200, body="Success")
                    await http_client.fetch("https://example.com/page")

    @pytest.mark.asyncio
    async def test_backoff_expiry_timing(self, http_client):
        """Test backoff registry expiry with precise timing."""
        registry = DomainBackoffRegistry(cooldown_seconds=0.5)  # Short cooldown

        # Record 3 failures to trigger cooldown
        await registry.record_failure("test.com")
        await registry.record_failure("test.com")
        await registry.record_failure("test.com")

        # Should be in cooldown
        assert await registry.is_in_cooldown("test.com")

        # Wait for expiry
        await asyncio.sleep(0.6)

        # Should no longer be in cooldown
        assert not await registry.is_in_cooldown("test.com")

    @pytest.mark.asyncio
    async def test_proxy_scheme_matching(self, http_client):
        """Test proxy selection with scheme matching."""
        os.environ["QUARRY_HTTP_PROXIES"] = "http://proxy1.com:8080,https://proxy2.com:8080"

        try:
            # Reinitialize client to pick up proxy config
            await http_client.close()

            config = Config()
            config.debug.test_mode = True
            client = HttpClient(config)
            await client.initialize()

            try:
                # Test HTTPS URL should prefer HTTPS proxy
                https_proxy = client._get_next_proxy("https://example.com/secure")
                assert https_proxy is not None and "https:" in https_proxy

                # Test HTTP URL should prefer HTTP proxy
                http_proxy = client._get_next_proxy("http://example.com/page")
                assert http_proxy is not None and "http:" in http_proxy

            finally:
                await client.close()
        finally:
            os.environ.pop("QUARRY_HTTP_PROXIES", None)

    @pytest.mark.asyncio
    async def test_empty_proxy_list_handling(self, http_client):
        """Test behavior with empty proxy configuration."""
        # Ensure no proxies configured
        http_client.proxies.clear()

        proxy = http_client._get_next_proxy("https://example.com/page")
        assert proxy is None

    @pytest.mark.asyncio
    async def test_retry_attempts_basic(self, http_client):
        """Test retry attempts work correctly with different configurations."""
        test_cases = [1, 2, 3]

        for max_retries in test_cases:
            with patch("random.uniform", return_value=1.0):  # Deterministic jitter
                call_count = 0

                def failing_callback(url, **kwargs):
                    nonlocal call_count
                    call_count += 1
                    return CallbackResult(status=503, body="Service Unavailable")

                with aioresponses() as m:
                    # Add multiple responses for retries
                    for _ in range(max_retries + 2):  # Add extra for safety
                        m.get("https://example.com/page", callback=failing_callback)

                    response = await http_client.fetch("https://example.com/page", max_retries=max_retries)

                    # Should attempt initial + retries
                    expected_attempts = max_retries + 1
                    assert response.attempts == expected_attempts

    @pytest.mark.asyncio
    async def test_backoff_timing_basic(self, http_client):
        """Test backoff timing with different cooldown periods."""
        test_cases = [0.1, 0.5, 1.0]

        for cooldown_seconds in test_cases:
            registry = DomainBackoffRegistry(cooldown_seconds=cooldown_seconds)

            # Trigger cooldown
            for _ in range(3):
                await registry.record_failure("test.com")

            start_time = time.time()
            assert await registry.is_in_cooldown("test.com")

            # Wait for cooldown to expire
            await asyncio.sleep(cooldown_seconds + 0.1)

            end_time = time.time()
            elapsed = end_time - start_time

            # Should no longer be in cooldown after the timeout
            assert not await registry.is_in_cooldown("test.com")
            assert elapsed >= cooldown_seconds

    @pytest.mark.asyncio
    async def test_fetch_without_initialization(self):
        """Test fetch behavior before client initialization."""
        config = Config()
        client = HttpClient(config)

        # Should raise RuntimeError when not initialized
        with pytest.raises(RuntimeError, match="HTTP client not initialized"):
            await client.fetch("https://example.com/page")

    @pytest.mark.asyncio
    async def test_close_without_session(self):
        """Test close behavior when session is None."""
        config = Config()
        client = HttpClient(config)

        # Should not raise exception
        await client.close()
        assert client.session is None

    @pytest.mark.asyncio
    async def test_robots_cache_property_lazy_loading(self, http_client):
        """Test robots cache lazy loading behavior."""
        # Before session initialization
        config = Config()
        client = HttpClient(config)

        # Should return None before session
        assert client.robots_cache is None

        await client.initialize()

        # Should return cache after session initialization
        assert client.robots_cache is not None

        await client.close()

    @pytest.mark.asyncio
    async def test_connection_error_with_proxy(self, http_client):
        """Test connection errors when using proxy."""
        os.environ["QUARRY_HTTP_PROXIES"] = "http://invalid-proxy:8080"

        try:
            config = Config()
            config.debug.test_mode = True
            client = HttpClient(config)
            await client.initialize()

            try:
                response = await client.fetch("https://example.com/page")

                # Should handle proxy connection failure gracefully
                assert response.status == 0
                assert response.attempts > 0

            finally:
                await client.close()
        finally:
            os.environ.pop("QUARRY_HTTP_PROXIES", None)

    @pytest.mark.asyncio
    async def test_response_encoding_edge_cases(self, http_client):
        """Test various response encoding scenarios."""
        test_cases = [
            (b"\xff\xfe\x00\x00", "Invalid UTF-8"),  # Invalid bytes
            (b"", "Empty response"),
            (b"\x00\x01\x02", "Binary data"),
        ]

        for body_bytes, _case_name in test_cases:
            with aioresponses() as m:
                m.get("https://example.com/page", status=200, body=body_bytes)

                response = await http_client.fetch("https://example.com/page")

                # Should handle all cases gracefully
                assert response.status == 200
                assert response.body == body_bytes
                assert response.attempts == 1

    @pytest.mark.asyncio
    async def test_stats_with_multiple_domains(self, http_client):
        """Test statistics collection across multiple domains."""
        domains = ["example.com", "test.com", "demo.org"]

        # Access semaphores for different domains
        for domain in domains:
            await http_client._get_domain_semaphore(domain)

        stats = http_client.get_stats()

        # Should track multiple domain semaphores
        assert stats["domain_semaphores"] >= len(domains)
        assert stats["in_flight_requests"] >= 0
        assert isinstance(stats["backoff_stats"], dict)

    @pytest.mark.asyncio
    async def test_timeout_with_custom_value(self, http_client):
        """Test custom timeout handling."""
        # Mock the _perform_request to raise TimeoutError directly
        original_perform_request = http_client._perform_request

        async def timeout_perform_request(url, timeout, proxy=None):
            # Simulate timeout by raising asyncio.TimeoutError
            raise asyncio.TimeoutError(f"Request timed out after {timeout}s")

        # Patch the method to simulate timeout
        http_client._perform_request = timeout_perform_request

        try:
            response = await http_client.fetch("https://example.com/page", timeout=0.1)  # Very short timeout

            # Should timeout and return error status
            assert response.status == 0  # Error status after all retries exhausted
            assert response.attempts > 1  # Should have retried
        finally:
            # Restore original method
            http_client._perform_request = original_perform_request
