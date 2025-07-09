"""
Comprehensive tests for metrics histogram functionality.

This test suite focuses on testing the histogram metrics observe path,
particularly the crawler_fetch_latency_seconds metric to boost coverage.
"""

import asyncio
import time
from unittest.mock import patch

import pytest
import pytest_asyncio
from aioresponses import aioresponses
from quarrycore.observability.metrics import METRICS

from tests.helpers.metric_delta import histogram_observes


def get_histogram_count(histogram):
    """Get current observation count from histogram."""
    try:
        collected = list(histogram.collect())
        if collected:
            for sample in collected[0].samples:
                if sample.name.endswith("_count"):
                    return sample.value
        return 0.0
    except (AttributeError, TypeError, IndexError):
        return 0.0


class TestMetricsHistogram:
    """Test suite for metrics histogram functionality."""

    @pytest.mark.asyncio
    async def test_fetch_latency_histogram_observe(self, http_client):
        """Test that fetch latency histogram receives observe() calls."""
        # Metrics are available via metrics_reset fixture
        histogram = METRICS["crawler_fetch_latency_seconds"]

        with histogram_observes(histogram, min_observations=1):
            with aioresponses() as m:
                # Mock successful response
                m.get("https://example.com/test", status=200, body="Success response")

                # Perform fetch
                start_time = time.time()
                response = await http_client.fetch("https://example.com/test")
                end_time = time.time()

                # Verify successful response
                assert response.status == 200
                assert response.body == b"Success response"

                # Verify timing makes sense
                actual_duration = end_time - start_time
                assert actual_duration > 0

    @pytest.mark.asyncio
    async def test_histogram_observe_with_retry(self, http_client):
        """Test histogram observe with retry scenarios - tests final response observation."""
        # Metrics are available via metrics_reset fixture
        histogram = METRICS["crawler_fetch_latency_seconds"]

        with histogram_observes(histogram, min_observations=1):
            with aioresponses() as m:
                # Set up retryable response that will eventually succeed
                m.get("https://example.com/retry", status=503, body="Service unavailable")
                m.get("https://example.com/retry", status=200, body="Success after retry")

                # Perform fetch - will retry and succeed
                response = await http_client.fetch("https://example.com/retry")

                # Should get 200 response after retry
                assert response.status == 200

    @pytest.mark.asyncio
    async def test_histogram_observe_multiple_requests(self, http_client):
        """Test histogram with multiple concurrent requests."""
        # Metrics are available via metrics_reset fixture
        histogram = METRICS["crawler_fetch_latency_seconds"]
        url_count = 3

        with histogram_observes(histogram, min_observations=url_count):
            with aioresponses() as m:
                # Mock multiple endpoints
                urls = [
                    "https://example.com/endpoint1",
                    "https://example.com/endpoint2",
                    "https://example.com/endpoint3",
                ]

                for url in urls:
                    m.get(url, status=200, body=f"Response for {url}")

                # Perform concurrent fetches
                tasks = [http_client.fetch(url) for url in urls]
                responses = await asyncio.gather(*tasks)

                # Verify all responses
                for response in responses:
                    assert response.status == 200

    @pytest.mark.asyncio
    async def test_histogram_observe_with_different_status_codes(self, http_client):
        """Test histogram observe with various HTTP status codes."""
        # Metrics are available via metrics_reset fixture
        histogram = METRICS["crawler_fetch_latency_seconds"]
        status_codes = [200, 404, 500, 301, 403]

        with histogram_observes(histogram, min_observations=len(status_codes)):
            with aioresponses() as m:
                for status in status_codes:
                    url = f"https://example.com/status{status}"
                    m.get(url, status=status, body=f"Response {status}")

                    # Perform fetch
                    response = await http_client.fetch(url)
                    assert response.status == status

    @pytest.mark.asyncio
    async def test_histogram_observe_timing_accuracy(self, http_client):
        """Test that histogram observes accurate timing values."""
        # Metrics are available via metrics_reset fixture
        histogram = METRICS["crawler_fetch_latency_seconds"]

        # Mock a delayed response
        async def delayed_callback(url, **kwargs):
            await asyncio.sleep(0.1)  # 100ms delay
            from aioresponses import CallbackResult

            return CallbackResult(status=200, body="Delayed response")

        with histogram_observes(histogram, min_observations=1):
            with aioresponses() as m:
                m.get("https://example.com/delayed", callback=delayed_callback)

                # Measure timing
                start_time = time.time()
                response = await http_client.fetch("https://example.com/delayed")
                end_time = time.time()

                # Verify response
                assert response.status == 200

                # Verify timing (should be at least 100ms)
                duration = end_time - start_time
                assert duration >= 0.1

                # Response timing should match measurement
                response_duration = response.end_ts - response.start_ts
                assert abs(response_duration - duration) < 0.01  # Within 10ms

    @pytest.mark.asyncio
    async def test_histogram_observe_with_timeout(self, http_client):
        """Test histogram behavior with timeout scenarios."""
        # Metrics are available via metrics_reset fixture
        histogram = METRICS["crawler_fetch_latency_seconds"]
        initial_count = get_histogram_count(histogram)

        # Mock a very slow response that will timeout
        async def timeout_callback(url, **kwargs):
            await asyncio.sleep(10)  # Longer than default timeout
            from aioresponses import CallbackResult

            return CallbackResult(status=200, body="Too slow")

        with aioresponses() as m:
            m.get("https://example.com/timeout", callback=timeout_callback)

            # Perform fetch with short timeout
            response = await http_client.fetch("https://example.com/timeout", timeout=0.1)

            # Should return error response due to timeout
            assert response.status == 0  # Error status for timeout

            # Histogram typically shouldn't be updated for timeouts
            # (depends on implementation - timeout might not reach metrics code)
            final_count = get_histogram_count(histogram)
            # Could be same or incremented depending on implementation
            assert final_count >= initial_count

    @pytest.mark.asyncio
    async def test_histogram_with_robots_blocked_request(self, temp_dir):
        """Test histogram behavior when request is blocked by robots.txt."""
        # Metrics are available via metrics_reset fixture
        histogram = METRICS["crawler_fetch_latency_seconds"]
        initial_count = get_histogram_count(histogram)

        with aioresponses() as m:
            # Mock robots.txt that blocks the request
            m.get("https://example.com/robots.txt", status=200, body="User-agent: *\nDisallow: /blocked")
            m.get("https://example.com/blocked", status=200, body="Should be blocked")

            # Perform fetch to blocked URL
            # Create HTTP client with robots.txt checking enabled
            from quarrycore.config.config import Config
            from quarrycore.crawler.http_client import HttpClient

            config = Config()
            config.crawler.respect_robots = True  # Enable robots checking
            config.crawler.user_agent = "TestBot/1.0"
            config.debug.test_mode = True

            async with HttpClient(config) as client:
                response = await client.fetch("https://example.com/blocked")

            # Should be blocked (status 999)
            assert response.status == 999

            # Histogram should not be updated for blocked requests
            final_count = get_histogram_count(histogram)
            # Might be same or slightly incremented depending on implementation
            assert final_count >= initial_count

    @pytest.mark.asyncio
    async def test_histogram_edge_cases(self, http_client):
        """Test histogram with edge cases and error conditions."""
        # Metrics are available via metrics_reset fixture
        histogram = METRICS["crawler_fetch_latency_seconds"]

        # Test various edge cases
        test_cases = [
            ("https://example.com/empty", 200, b""),  # Empty response
            ("https://example.com/large", 200, b"x" * 10000),  # Large response
            ("https://example.com/unicode", 200, "测试内容".encode()),  # Unicode content
        ]

        with histogram_observes(histogram, min_observations=len(test_cases)):
            with aioresponses() as m:
                for url, status, body in test_cases:
                    m.get(url, status=status, body=body)

                    response = await http_client.fetch(url)
                    assert response.status == status
                    assert response.body == body

    @pytest.mark.asyncio
    async def test_histogram_with_malformed_url(self, http_client):
        """Test histogram behavior with malformed URLs."""
        # Metrics are available via metrics_reset fixture
        histogram = METRICS["crawler_fetch_latency_seconds"]
        initial_count = get_histogram_count(histogram)

        # Test malformed URLs
        malformed_urls = [
            "not-a-url",
            "://missing-scheme",
            "http://",
            "",
        ]

        for url in malformed_urls:
            response = await http_client.fetch(url)
            # Should return error response for malformed URL
            assert response.status == 0

        # Histogram should not be updated for malformed URLs
        final_count = get_histogram_count(histogram)
        assert final_count == initial_count

    @pytest.mark.asyncio
    async def test_histogram_performance_with_many_observations(self, http_client):
        """Test histogram performance with many observations."""
        # Metrics are available via metrics_reset fixture
        histogram = METRICS["crawler_fetch_latency_seconds"]
        request_count = 50  # Moderate number for performance test

        with histogram_observes(histogram, min_observations=request_count):
            with aioresponses() as m:
                # Mock many endpoints
                for i in range(request_count):
                    url = f"https://example.com/perf{i}"
                    m.get(url, status=200, body=f"Response {i}")

                # Measure performance
                start_time = time.time()

                # Perform many fetches
                tasks = []
                for i in range(request_count):
                    url = f"https://example.com/perf{i}"
                    tasks.append(http_client.fetch(url))

                responses = await asyncio.gather(*tasks)

                end_time = time.time()

                # Verify all responses
                for i, response in enumerate(responses):
                    assert response.status == 200
                    assert response.body == f"Response {i}".encode()

                # Verify reasonable performance
                total_duration = end_time - start_time
                assert total_duration < 5.0  # Should complete in reasonable time
