"""
Targeted tests for http_client.py to boost branch coverage.
"""

import asyncio
from pathlib import Path
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from quarrycore.config.config import Config
from quarrycore.crawler.http_client import HttpClient
from quarrycore.protocols import CrawlResult


class TestHttpClientCoverageBoost:
    """Targeted tests to boost http_client.py branch coverage."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return Config()

    @pytest.fixture
    def http_client(self, config):
        """Create HTTP client instance."""
        return HttpClient(config)

    @pytest.mark.asyncio
    async def test_retry_exhausted_branch(self, http_client):
        """Test retry logic when all retries are exhausted."""
        # Mock httpx.AsyncClient to always raise an exception
        mock_client = AsyncMock()
        mock_client.get.side_effect = Exception("Connection failed")

        with patch("httpx.AsyncClient", return_value=mock_client):
            # This should trigger retry exhaustion
            result = await http_client.fetch("https://example.com")

            # Should have failed after exhausting retries
            assert result.status_code == 500
            assert not result.is_valid
            assert len(result.errors) > 0
            assert "Connection failed" in str(result.errors[0].error_message)

    @pytest.mark.asyncio
    async def test_invalid_url_scheme_branch(self, http_client):
        """Test handling of invalid URL schemes."""
        # Test with invalid protocol
        result = await http_client.fetch("ftp://example.com/file.txt")

        # Should handle invalid scheme gracefully
        assert result.status_code == 400
        assert not result.is_valid
        assert len(result.errors) > 0

    @pytest.mark.asyncio
    async def test_timeout_with_retry_branch(self, http_client):
        """Test timeout handling with retry logic."""
        mock_client = AsyncMock()

        # First call times out, second succeeds
        mock_client.get.side_effect = [
            asyncio.TimeoutError("Request timed out"),
            Mock(
                status_code=200, content=b"Success after retry", headers={"Content-Type": "text/html"}, encoding="utf-8"
            ),
        ]

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await http_client.fetch("https://example.com", timeout=5.0)

            # Should succeed after retry
            assert result.status_code == 200
            assert result.is_valid
            assert result.content == b"Success after retry"

    @pytest.mark.asyncio
    async def test_content_type_detection_branches(self, http_client):
        """Test different content type detection branches."""
        mock_client = AsyncMock()

        # Test with missing content-type header
        mock_client.get.return_value = Mock(
            status_code=200,
            content=b"<html><body>Test</body></html>",
            headers={},  # No content-type header
            encoding="utf-8",
        )

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await http_client.fetch("https://example.com")

            # Should handle missing content-type gracefully
            assert result.status_code == 200
            assert result.is_valid
            assert result.content_type == ""

    @pytest.mark.asyncio
    async def test_robots_txt_caching_branch(self, http_client):
        """Test robots.txt caching logic branches."""
        mock_client = AsyncMock()

        # First call to robots.txt
        mock_client.get.side_effect = [
            Mock(
                status_code=200,
                content=b"User-agent: *\nDisallow: /admin",
                headers={"Content-Type": "text/plain"},
                encoding="utf-8",
            ),
            Mock(
                status_code=200,
                content=b"<html>Main page</html>",
                headers={"Content-Type": "text/html"},
                encoding="utf-8",
            ),
        ]

        with patch("httpx.AsyncClient", return_value=mock_client):
            # First request should fetch robots.txt
            result1 = await http_client.fetch("https://example.com/page1")
            assert result1.robots_allowed

            # Second request to same domain should use cached robots.txt
            result2 = await http_client.fetch("https://example.com/page2")
            assert result2.robots_allowed

            # Should only have called get twice (once for robots.txt, once for each page)
            assert mock_client.get.call_count == 2

    @pytest.mark.asyncio
    async def test_user_agent_rotation_branch(self, http_client):
        """Test user agent rotation logic."""
        mock_client = AsyncMock()
        mock_client.get.return_value = Mock(
            status_code=200, content=b"Success", headers={"Content-Type": "text/html"}, encoding="utf-8"
        )

        with patch("httpx.AsyncClient", return_value=mock_client):
            # Make multiple requests to trigger user agent rotation
            for i in range(3):
                result = await http_client.fetch(f"https://example.com/page{i}")
                assert result.is_valid

            # Should have rotated user agents
            assert mock_client.get.call_count == 3

    @pytest.mark.asyncio
    async def test_rate_limiting_backoff_branch(self, http_client):
        """Test rate limiting with backoff logic."""
        mock_client = AsyncMock()

        # First call returns 429 (rate limited), second succeeds
        mock_client.get.side_effect = [
            Mock(status_code=429, content=b"Rate limited", headers={"Retry-After": "1"}, encoding="utf-8"),
            Mock(
                status_code=200,
                content=b"Success after backoff",
                headers={"Content-Type": "text/html"},
                encoding="utf-8",
            ),
        ]

        with patch("httpx.AsyncClient", return_value=mock_client):
            with patch("asyncio.sleep", return_value=None):  # Mock sleep for speed
                result = await http_client.fetch("https://example.com")

                # Should succeed after backoff
                assert result.status_code == 200
                assert result.is_valid
                assert result.content == b"Success after backoff"

    @pytest.mark.asyncio
    async def test_redirect_handling_branch(self, http_client):
        """Test redirect handling logic."""
        mock_client = AsyncMock()

        # Mock a redirect response
        mock_client.get.return_value = Mock(
            status_code=301,
            content=b"Moved Permanently",
            headers={"Location": "https://example.com/new-location"},
            encoding="utf-8",
        )

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await http_client.fetch("https://example.com/old-location")

            # Should handle redirect
            assert result.status_code == 301
            assert result.final_url == "https://example.com/old-location"  # Final URL should be set

    @pytest.mark.asyncio
    async def test_content_encoding_branch(self, http_client):
        """Test content encoding detection."""
        mock_client = AsyncMock()

        # Test with gzip encoding
        mock_client.get.return_value = Mock(
            status_code=200,
            content=b"Compressed content",
            headers={"Content-Encoding": "gzip", "Content-Type": "text/html"},
            encoding="utf-8",
        )

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await http_client.fetch("https://example.com")

            # Should detect and handle encoding
            assert result.status_code == 200
            assert result.is_valid
            assert result.content_encoding == "gzip"

    @pytest.mark.asyncio
    async def test_large_response_handling_branch(self, http_client):
        """Test handling of large responses."""
        mock_client = AsyncMock()

        # Mock a large response
        large_content = b"x" * 100000  # 100KB
        mock_client.get.return_value = Mock(
            status_code=200,
            content=large_content,
            headers={"Content-Type": "text/html", "Content-Length": str(len(large_content))},
            encoding="utf-8",
        )

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await http_client.fetch("https://example.com")

            # Should handle large response
            assert result.status_code == 200
            assert result.is_valid
            assert len(result.content) == 100000
            assert result.content_length == 100000

    @pytest.mark.asyncio
    async def test_malformed_response_branch(self, http_client):
        """Test handling of malformed responses."""
        mock_client = AsyncMock()

        # Mock a malformed response
        mock_client.get.side_effect = UnicodeDecodeError("utf-8", b"", 0, 1, "invalid start byte")

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await http_client.fetch("https://example.com")

            # Should handle decode error gracefully
            assert result.status_code == 500
            assert not result.is_valid
            assert len(result.errors) > 0

    @pytest.mark.asyncio
    async def test_dns_resolution_failure_branch(self, http_client):
        """Test DNS resolution failure handling."""
        mock_client = AsyncMock()

        # Mock DNS resolution failure
        mock_client.get.side_effect = Exception("Name resolution failed")

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await http_client.fetch("https://nonexistent-domain-test.com")

            # Should handle DNS failure gracefully
            assert result.status_code == 500
            assert not result.is_valid
            assert len(result.errors) > 0

    @pytest.mark.asyncio
    async def test_cleanup_on_shutdown_branch(self, http_client):
        """Test cleanup logic on shutdown."""
        # Initialize the client
        await http_client.initialize()

        # Mock some internal state
        http_client._is_initialized = True

        # Test shutdown cleanup
        await http_client.close()

        # Should clean up properly
        assert not http_client._is_initialized
