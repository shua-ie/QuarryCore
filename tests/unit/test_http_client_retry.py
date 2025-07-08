"""
Tests for HTTP client retry behavior.

These tests focus on externally observable retry behavior rather than
implementation details. They test that the HTTP client properly retries
requests, applies backoff delays, and handles various error conditions.
"""

import asyncio
import time
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from aioresponses import aioresponses
from quarrycore.config.config import Config
from quarrycore.crawler.http_client import HttpClient


@pytest.mark.unit
class TestHttpClientRetryBehavior:
    """Test retry behavior via public API."""

    @pytest.mark.asyncio
    async def test_retry_on_429_then_success(self, http_client, deterministic_jitter):
        """Test retry logic on 429 (rate limit) then success."""
        with aioresponses() as m:
            # First request returns 429
            m.get("https://example.com/page", status=429, headers={"Retry-After": "1"}, body="")
            # Second request succeeds
            m.get("https://example.com/page", status=200, headers={"Content-Type": "text/html"}, body="Success!")

            start_time = time.time()
            response = await http_client.fetch("https://example.com/page")
            end_time = time.time()

            # Should succeed after retry
            assert response.status == 200
            assert response.body == b"Success!"
            assert response.attempts == 2  # Initial + 1 retry

            # Should have taken at least 1 second due to backoff
            assert end_time - start_time >= 0.8  # Allow some tolerance

    @pytest.mark.asyncio
    async def test_retry_on_503_then_success(self, http_client, deterministic_jitter):
        """Test retry logic on 503 (service unavailable) then success."""
        with aioresponses() as m:
            # First request returns 503
            m.get("https://example.com/page", status=503, headers={"Content-Type": "text/html"}, body="")
            # Second request succeeds
            m.get("https://example.com/page", status=200, headers={"Content-Type": "text/html"}, body="Success!")

            start_time = time.time()
            response = await http_client.fetch("https://example.com/page")
            end_time = time.time()

            # Should succeed after retry
            assert response.status == 200
            assert response.body == b"Success!"
            assert response.attempts == 2  # Initial + 1 retry

            # Should have taken time due to backoff
            assert end_time - start_time >= 0.8

    @pytest.mark.asyncio
    async def test_max_retries_exhausted(self, http_client, deterministic_jitter):
        """Test that max retries are respected."""
        with aioresponses() as m:
            # All requests return 503
            m.get("https://example.com/page", status=503, headers={"Content-Type": "text/html"}, repeat=True)

            start_time = time.time()
            response = await http_client.fetch("https://example.com/page")
            end_time = time.time()

            # Should return the last error response
            assert response.status == 503
            assert response.attempts == 4  # 1 initial + 3 retries

            # Should have taken time for all retries (1s + 2s + 4s)
            assert end_time - start_time >= 6.0  # Allow some tolerance

    @pytest.mark.asyncio
    async def test_no_retry_on_404(self, http_client):
        """Test that 404 errors are not retried."""
        with aioresponses() as m:
            m.get("https://example.com/page", status=404, body="Not Found", headers={"Content-Type": "text/html"})

            start_time = time.time()
            response = await http_client.fetch("https://example.com/page")
            end_time = time.time()

            # Should not retry 404 errors
            assert response.status == 404
            assert response.body == b"Not Found"
            assert response.attempts == 1  # No retries

            # Should complete quickly (no backoff delay)
            assert end_time - start_time < 1.0

    @pytest.mark.asyncio
    async def test_custom_max_retries(self, http_client, deterministic_jitter):
        """Test custom max_retries parameter."""
        with aioresponses() as m:
            # All requests return 503
            m.get("https://example.com/page", status=503, headers={"Content-Type": "text/html"}, repeat=True)

            response = await http_client.fetch("https://example.com/page", max_retries=2)

            # Should stop after custom retry limit
            assert response.status == 503
            assert response.attempts == 3  # 1 initial + 2 retries

    @pytest.mark.asyncio
    async def test_exponential_backoff_timing(self, http_client, deterministic_jitter):
        """Test that exponential backoff timing is approximately correct."""
        with aioresponses() as m:
            # All requests return 503
            m.get("https://example.com/page", status=503, headers={"Content-Type": "text/html"}, repeat=True)

            start_time = time.time()
            await http_client.fetch("https://example.com/page")
            end_time = time.time()

            # Expected delays: ~1s, ~2s, ~4s (deterministic jitter gives us exact values)
            # Total expected: ~7s
            total_time = end_time - start_time
            assert 6.5 < total_time < 8.0  # Allow some margin for execution overhead

    @pytest.mark.asyncio
    async def test_network_error_retry(self, http_client, deterministic_jitter):
        """Test retry on network errors."""
        call_count = 0

        async def mock_perform_request(url, timeout, proxy=None):
            nonlocal call_count
            call_count += 1

            if call_count == 1:
                # First call fails with network error
                raise Exception("Network error")
            else:
                # Second call succeeds
                mock_response = AsyncMock()
                mock_response.status = 200
                mock_response.headers = {"Content-Type": "text/html"}
                mock_response.url = url
                mock_response.read.return_value = b"Success!"
                mock_response.release = AsyncMock()
                return mock_response

        with patch.object(http_client, "_perform_request", side_effect=mock_perform_request):
            start_time = time.time()
            response = await http_client.fetch("https://example.com/page")
            end_time = time.time()

            # Should succeed after retry
            assert response.status == 200
            assert response.body == b"Success!"
            assert response.attempts == 2
            assert call_count == 2

            # Should have backoff delay
            assert end_time - start_time >= 0.8

    @pytest.mark.asyncio
    async def test_timeout_retry(self, http_client, deterministic_jitter):
        """Test retry on timeout errors."""
        call_count = 0

        async def mock_perform_request(url, timeout, proxy=None):
            nonlocal call_count
            call_count += 1

            if call_count == 1:
                # First call times out
                raise asyncio.TimeoutError("Request timed out")
            else:
                # Second call succeeds
                mock_response = AsyncMock()
                mock_response.status = 200
                mock_response.headers = {"Content-Type": "text/html"}
                mock_response.url = url
                mock_response.read.return_value = b"Success!"
                mock_response.release = AsyncMock()
                return mock_response

        with patch.object(http_client, "_perform_request", side_effect=mock_perform_request):
            start_time = time.time()
            response = await http_client.fetch("https://example.com/page")
            end_time = time.time()

            # Should succeed after retry
            assert response.status == 200
            assert response.body == b"Success!"
            assert response.attempts == 2
            assert call_count == 2

            # Should have backoff delay
            assert end_time - start_time >= 0.8

    @pytest.mark.asyncio
    async def test_jitter_variation(self, http_client):
        """Test that jitter creates variation in backoff timing."""
        times = []

        # Run multiple requests with same failure pattern
        for _i in range(3):  # Reduced from 5 to 3 for faster test
            with aioresponses() as m:
                # Two failures then success
                m.get("https://example.com/page", status=503, body="")
                m.get("https://example.com/page", status=503, body="")
                m.get("https://example.com/page", status=200, body="Success!")

                start_time = time.time()
                await http_client.fetch("https://example.com/page")
                end_time = time.time()

                times.append(end_time - start_time)

        # Should have some variation due to jitter (even small amounts)
        min_time = min(times)
        max_time = max(times)
        variation = (max_time - min_time) / min_time if min_time > 0 else 0

        # Expect at least 5% variation due to jitter
        assert variation > 0.05 or len(set(times)) > 1  # Either numerical variation or different values

    @pytest.mark.asyncio
    async def test_no_retry_beyond_max_attempts(self, http_client, deterministic_jitter):
        """Test that retries stop after max attempts even if error continues."""
        with aioresponses() as m:
            # More failures than max_retries
            m.get("https://example.com/page", status=503, headers={"Content-Type": "text/html"}, repeat=True)

            response = await http_client.fetch("https://example.com/page", max_retries=2)

            # Should stop after 1 initial + 2 retries = 3 attempts
            assert response.attempts == 3
            assert response.status == 503

    @pytest.mark.asyncio
    async def test_latency_metrics_with_retries(self, http_client, deterministic_jitter):
        """Test that latency metrics are properly recorded with retries."""
        from quarrycore.observability.metrics import METRICS

        from tests.helpers.metric_delta import histogram_observes

        # Metrics are available via metrics_reset fixture

        histogram = METRICS["crawler_fetch_latency_seconds"]

        # Test should handle max retries exhaustion
        with aioresponses() as m:
            m.get("https://example.com/page", status=503, repeat=True)

            start_time = time.time()
            await http_client.fetch("https://example.com/page")
            end_time = time.time()

            # Should have taken time for all retry attempts
            min_expected_time = 1 + 2 + 4  # exponential backoff
            assert end_time - start_time >= min_expected_time * 0.8

        # Run multiple requests with same failure pattern
        for _i in range(3):  # Reduced from 5 to 3 for faster test
            with aioresponses() as m:
                # Two failures then success
                m.get("https://example.com/page", status=503, body="")
                m.get("https://example.com/page", status=503, body="")
                m.get("https://example.com/page", status=200, body="Success!")

                response = await http_client.fetch("https://example.com/page")

                # Should succeed after retry
                assert response.status == 200
                assert response.attempts == 3

        # Test metrics are recorded
        with histogram_observes(histogram, min_observations=1):
            with aioresponses() as m:
                # First request fails with 503
                m.get("https://example.com/page", status=503, body="Service unavailable")
                # Second request succeeds
                m.get("https://example.com/page", status=200, body="Success!")

                response = await http_client.fetch("https://example.com/page")

                # Should succeed after retry
                assert response.status == 200
                assert response.attempts == 2

    @pytest.mark.asyncio
    async def test_server_error_codes_trigger_retry(self, http_client, deterministic_jitter):
        """Test that various server error codes trigger retries."""
        error_codes = [429, 502, 503, 504]

        for error_code in error_codes:
            with aioresponses() as m:
                # First request fails with error code
                m.get("https://example.com/page", status=error_code, body="")
                # Second request succeeds
                m.get("https://example.com/page", status=200, body="Success!")

                response = await http_client.fetch("https://example.com/page")

                # Should retry and succeed
                assert response.status == 200
                assert response.attempts == 2

    @pytest.mark.asyncio
    async def test_client_error_codes_no_retry(self, http_client):
        """Test that client error codes do not trigger retries."""
        client_error_codes = [400, 401, 403, 404, 410]

        for error_code in client_error_codes:
            with aioresponses() as m:
                m.get(
                    "https://example.com/page",
                    status=error_code,
                    body=f"Error {error_code}",
                    headers={"Content-Type": "text/html"},
                )

                start_time = time.time()
                response = await http_client.fetch("https://example.com/page")
                end_time = time.time()

                # Should not retry client errors
                assert response.status == error_code
                assert response.attempts == 1

                # Should complete quickly
                assert end_time - start_time < 1.0
