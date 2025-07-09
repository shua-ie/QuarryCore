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
    @pytest.mark.asyncio
    async def http_client(self, config):
        """Create HTTP client instance."""
        client = HttpClient(config)
        await client.initialize()
        yield client
        await client.close()

    @pytest.mark.asyncio
    async def test_retry_exhausted_branch(self, config):
        """Test retry logic when all retries are exhausted."""
        client = HttpClient(config)
        await client.initialize()

        try:
            # Mock aiohttp.ClientSession to always raise an exception
            with patch("aiohttp.ClientSession") as mock_session_class:
                mock_session = AsyncMock()
                mock_session.get.side_effect = Exception("Connection failed")
                mock_session_class.return_value = mock_session

                # Replace the session directly
                client.session = mock_session

                # This should trigger retry exhaustion
                result = await client.fetch("https://example.com")

                # Should have failed after exhausting retries
                assert result.status == 0  # 0 indicates failure
                assert result.body == b""
                assert result.attempts > 0
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_invalid_url_scheme_branch(self, config):
        """Test handling of invalid URL schemes."""
        client = HttpClient(config)
        await client.initialize()

        try:
            # Test with invalid protocol - this should be caught before making any request
            result = await client.fetch("ftp://example.com/file.txt")

            # Should handle invalid scheme gracefully
            assert result.status == 0  # 0 indicates failure
            assert result.body == b""
            assert result.attempts > 0
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_timeout_with_retry_branch(self, config):
        """Test timeout handling with retry logic."""
        client = HttpClient(config)
        await client.initialize()

        try:
            # Mock the session directly
            mock_session = AsyncMock()

            # First call times out, second succeeds
            mock_response = Mock()
            mock_response.status = 200
            mock_response.headers = {"Content-Type": "text/html"}
            mock_response.url = "https://example.com"
            mock_response.read = AsyncMock(return_value=b"Success after retry")
            mock_response.release = AsyncMock()

            mock_session.get.side_effect = [asyncio.TimeoutError("Request timed out"), mock_response]

            # Replace the session
            client.session = mock_session

            result = await client.fetch("https://example.com", timeout=5.0)

            # Should succeed after retry
            assert result.status == 200
            assert result.body == b"Success after retry"
            assert result.attempts == 2
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_content_type_detection_branches(self, config):
        """Test different content type detection branches."""
        client = HttpClient(config)
        await client.initialize()

        try:
            # Mock the session directly
            mock_session = AsyncMock()

            # Test with missing content-type header
            mock_response = Mock()
            mock_response.status = 200
            mock_response.headers = {}  # No content-type header
            mock_response.url = "https://example.com"
            mock_response.read = AsyncMock(return_value=b"<html><body>Test</body></html>")
            mock_response.release = AsyncMock()

            mock_session.get.return_value = mock_response

            # Replace the session
            client.session = mock_session

            result = await client.fetch("https://example.com")

            # Should handle missing content-type gracefully
            assert result.status == 200
            assert result.body == b"<html><body>Test</body></html>"
            assert result.headers == {}
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_robots_txt_caching_branch(self, config):
        """Test robots.txt caching logic branches."""
        client = HttpClient(config)
        await client.initialize()

        try:
            # Mock the session directly
            mock_session = AsyncMock()

            # Create mock responses
            robots_response = Mock()
            robots_response.status = 200
            robots_response.headers = {"Content-Type": "text/plain"}
            robots_response.url = "https://example.com/robots.txt"
            robots_response.read = AsyncMock(return_value=b"User-agent: *\nDisallow: /admin")
            robots_response.release = AsyncMock()

            page_response1 = Mock()
            page_response1.status = 200
            page_response1.headers = {"Content-Type": "text/html"}
            page_response1.url = "https://example.com/page1"
            page_response1.read = AsyncMock(return_value=b"<html>Main page 1</html>")
            page_response1.release = AsyncMock()

            page_response2 = Mock()
            page_response2.status = 200
            page_response2.headers = {"Content-Type": "text/html"}
            page_response2.url = "https://example.com/page2"
            page_response2.read = AsyncMock(return_value=b"<html>Main page 2</html>")
            page_response2.release = AsyncMock()

            # First call fetches robots.txt, then page1
            # Second call should use cached robots.txt and fetch page2 directly
            mock_session.get.side_effect = [robots_response, page_response1, page_response2]

            # Replace the session
            client.session = mock_session

            # First request should fetch robots.txt
            result1 = await client.fetch("https://example.com/page1")
            assert result1.status == 200

            # Second request to same domain should use cached robots.txt
            result2 = await client.fetch("https://example.com/page2")
            assert result2.status == 200

            # Should have called get twice (robots.txt+page1, then page2)
            # The robots.txt is cached, so second request only fetches page2
            assert mock_session.get.call_count == 2
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_user_agent_rotation_branch(self, config):
        """Test user agent rotation logic."""
        client = HttpClient(config)
        await client.initialize()

        try:
            # Mock the session directly
            mock_session = AsyncMock()

            def create_response():
                response = Mock()
                response.status = 200
                response.headers = {"Content-Type": "text/html"}
                response.url = "https://example.com"
                response.read = AsyncMock(return_value=b"Success")
                response.release = AsyncMock()
                return response

            mock_session.get.side_effect = [create_response(), create_response(), create_response()]

            # Replace the session
            client.session = mock_session

            # Make multiple requests to trigger user agent rotation
            for i in range(3):
                result = await client.fetch(f"https://example.com/page{i}")
                assert result.status == 200

            # Should have rotated user agents
            assert mock_session.get.call_count == 3
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_rate_limiting_backoff_branch(self, config):
        """Test rate limiting with backoff logic."""
        client = HttpClient(config)
        await client.initialize()

        try:
            # Mock the session directly
            mock_session = AsyncMock()

            # First call returns 429 (rate limited)
            rate_limited_response = Mock()
            rate_limited_response.status = 429
            rate_limited_response.headers = {"Retry-After": "1"}
            rate_limited_response.url = "https://example.com"
            rate_limited_response.read = AsyncMock(return_value=b"Rate limited")
            rate_limited_response.release = AsyncMock()

            # Second call succeeds
            success_response = Mock()
            success_response.status = 200
            success_response.headers = {"Content-Type": "text/html"}
            success_response.url = "https://example.com"
            success_response.read = AsyncMock(return_value=b"Success after backoff")
            success_response.release = AsyncMock()

            mock_session.get.side_effect = [rate_limited_response, success_response]

            # Replace the session
            client.session = mock_session

            with patch("asyncio.sleep", return_value=None):  # Mock sleep for speed
                result = await client.fetch("https://example.com")

                # Should succeed after backoff
                assert result.status == 200
                assert result.body == b"Success after backoff"
                assert result.attempts == 2
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_redirect_handling_branch(self, config):
        """Test redirect handling logic."""
        client = HttpClient(config)
        await client.initialize()

        try:
            # Mock the session directly
            mock_session = AsyncMock()

            # Mock a redirect response
            redirect_response = Mock()
            redirect_response.status = 301
            redirect_response.headers = {"Location": "https://example.com/new-location"}
            redirect_response.url = "https://example.com/old-location"
            redirect_response.read = AsyncMock(return_value=b"Moved Permanently")
            redirect_response.release = AsyncMock()

            mock_session.get.return_value = redirect_response

            # Replace the session
            client.session = mock_session

            result = await client.fetch("https://example.com/old-location")

            # Should handle redirect
            assert result.status == 301
            assert result.final_url == "https://example.com/old-location"
            assert result.body == b"Moved Permanently"
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_content_encoding_branch(self, config):
        """Test content encoding detection."""
        client = HttpClient(config)
        await client.initialize()

        try:
            # Mock the session directly
            mock_session = AsyncMock()

            # Test with gzip encoding
            gzip_response = Mock()
            gzip_response.status = 200
            gzip_response.headers = {"Content-Encoding": "gzip", "Content-Type": "text/html"}
            gzip_response.url = "https://example.com"
            gzip_response.read = AsyncMock(return_value=b"Compressed content")
            gzip_response.release = AsyncMock()

            mock_session.get.return_value = gzip_response

            # Replace the session
            client.session = mock_session

            result = await client.fetch("https://example.com")

            # Should detect and handle encoding
            assert result.status == 200
            assert result.body == b"Compressed content"
            assert result.headers["Content-Encoding"] == "gzip"
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_large_response_handling_branch(self, config):
        """Test handling of large responses."""
        client = HttpClient(config)
        await client.initialize()

        try:
            # Mock the session directly
            mock_session = AsyncMock()

            # Mock a large response
            large_content = b"x" * 100000  # 100KB
            large_response = Mock()
            large_response.status = 200
            large_response.headers = {"Content-Type": "text/html", "Content-Length": str(len(large_content))}
            large_response.url = "https://example.com"
            large_response.read = AsyncMock(return_value=large_content)
            large_response.release = AsyncMock()

            mock_session.get.return_value = large_response

            # Replace the session
            client.session = mock_session

            result = await client.fetch("https://example.com")

            # Should handle large response
            assert result.status == 200
            assert len(result.body) == 100000
            assert result.headers["Content-Length"] == "100000"
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_malformed_response_branch(self, config):
        """Test handling of malformed responses."""
        client = HttpClient(config)
        await client.initialize()

        try:
            # Mock the session directly
            mock_session = AsyncMock()

            # Mock a malformed response
            mock_session.get.side_effect = UnicodeDecodeError("utf-8", b"", 0, 1, "invalid start byte")

            # Replace the session
            client.session = mock_session

            result = await client.fetch("https://example.com")

            # Should handle decode error gracefully
            assert result.status == 0  # 0 indicates failure
            assert result.body == b""
            assert result.attempts > 0
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_dns_resolution_failure_branch(self, config):
        """Test DNS resolution failure handling."""
        client = HttpClient(config)
        await client.initialize()

        try:
            # Mock the session directly
            mock_session = AsyncMock()

            # Mock DNS resolution failure
            mock_session.get.side_effect = Exception("Name resolution failed")

            # Replace the session
            client.session = mock_session

            result = await client.fetch("https://nonexistent-domain-test.com")

            # Should handle DNS failure gracefully
            assert result.status == 0  # 0 indicates failure
            assert result.body == b""
            assert result.attempts > 0
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_cleanup_on_shutdown_branch(self, config):
        """Test cleanup logic on shutdown."""
        client = HttpClient(config)

        # Initialize the client
        await client.initialize()

        # Mock some internal state
        client._is_initialized = True

        # Test shutdown cleanup
        await client.close()

        # Should clean up properly
        assert not client._is_initialized
