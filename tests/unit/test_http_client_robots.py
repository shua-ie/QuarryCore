"""
Tests for HTTP client robots.txt compliance behavior.

These tests focus on externally observable behavior rather than
implementation details. They test that the HTTP client properly
respects robots.txt and meta robots directives by checking the
returned status codes and responses.
"""

import os

import pytest
import pytest_asyncio
from aioresponses import aioresponses
from quarrycore.config.config import Config
from quarrycore.crawler.http_client import HttpClient


@pytest.mark.unit
class TestHttpClientRobotsBehavior:
    """Test robots.txt compliance behavior via public API."""

    @pytest_asyncio.fixture
    async def robots_enabled_client(self, temp_dir):
        """Create HTTP client with robots.txt checking enabled."""
        from quarrycore.config.config import Config
        from quarrycore.crawler.http_client import HttpClient

        config = Config()
        config.crawler.max_retries = 3
        config.crawler.timeout = 5.0
        config.crawler.respect_robots = True  # Enable robots checking
        config.crawler.user_agent = "TestBot/1.0"
        config.debug.test_mode = True

        async with HttpClient(config) as client:
            yield client

    @pytest.mark.asyncio
    async def test_robots_txt_allows_crawling(self, robots_enabled_client):
        """Test that robots.txt allows crawling returns normal response."""
        with aioresponses() as m:
            # Mock robots.txt that allows all
            m.get("https://example.com/robots.txt", status=200, body="User-agent: *\nAllow: /")

            # Mock the actual page
            m.get(
                "https://example.com/page",
                status=200,
                body="<html><body>Test content</body></html>",
                headers={"Content-Type": "text/html"},
            )

            response = await robots_enabled_client.fetch("https://example.com/page")

            # Should get normal successful response
            assert response.status == 200
            assert response.body == b"<html><body>Test content</body></html>"
            assert response.attempts == 1

    @pytest.mark.asyncio
    async def test_robots_txt_blocks_crawling(self, robots_enabled_client):
        """Test that robots.txt blocks crawling returns status 999."""
        with aioresponses() as m:
            # Mock robots.txt that disallows all
            m.get("https://example.com/robots.txt", status=200, body="User-agent: *\nDisallow: /")

            response = await robots_enabled_client.fetch("https://example.com/page")

            # Should be blocked by robots.txt (status 999)
            assert response.status == 999
            assert response.body == b""
            assert response.attempts == 0

    @pytest.mark.asyncio
    async def test_robots_txt_disabled_allows_all(self):
        """Test that disabling robots.txt allows all crawling."""
        config = Config()
        config.crawler.respect_robots = False
        config.crawler.user_agent = "TestBot/1.0"

        client = HttpClient(config)
        await client.initialize()

        try:
            with aioresponses() as m:
                # Don't mock robots.txt - it shouldn't be checked
                m.get(
                    "https://example.com/page",
                    status=200,
                    body="<html><body>Test content</body></html>",
                    headers={"Content-Type": "text/html"},
                )

                response = await client.fetch("https://example.com/page")

                # Should get successful response without checking robots.txt
                assert response.status == 200
                assert response.body == b"<html><body>Test content</body></html>"
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_meta_robots_noindex_blocks_content(self, robots_enabled_client):
        """Test that meta robots noindex directive blocks content."""
        with aioresponses() as m:
            # Mock robots.txt that allows crawling
            m.get("https://example.com/robots.txt", status=200, body="User-agent: *\nAllow: /")

            # Mock page with meta robots noindex
            m.get(
                "https://example.com/page",
                status=200,
                body='<html><head><meta name="robots" content="noindex"></head><body>Test content</body></html>',
                headers={"Content-Type": "text/html"},
            )

            response = await robots_enabled_client.fetch("https://example.com/page")

            # Should be blocked by meta robots (status 999)
            assert response.status == 999
            assert response.body == b""
            assert response.attempts == 1  # Attempt was made but content blocked

    @pytest.mark.asyncio
    async def test_meta_robots_allows_indexing(self, robots_enabled_client):
        """Test that meta robots allows indexing when not restricted."""
        with aioresponses() as m:
            # Mock robots.txt that allows crawling
            m.get("https://example.com/robots.txt", status=200, body="User-agent: *\nAllow: /")

            # Mock page with meta robots allowing indexing
            m.get(
                "https://example.com/page",
                status=200,
                body='<html><head><meta name="robots" content="index,follow"></head><body>Test content</body></html>',
                headers={"Content-Type": "text/html"},
            )

            response = await robots_enabled_client.fetch("https://example.com/page")

            # Should get normal successful response
            assert response.status == 200
            assert b"Test content" in response.body
            assert response.attempts == 1

    @pytest.mark.asyncio
    async def test_no_meta_robots_tag_allows_crawling(self, http_client):
        """Test that absence of meta robots tag allows crawling."""
        with aioresponses() as m:
            # Mock robots.txt that allows crawling
            m.get("https://example.com/robots.txt", status=200, body="User-agent: *\nAllow: /")

            # Mock page without meta robots tag
            m.get(
                "https://example.com/page",
                status=200,
                body="<html><head><title>Test</title></head><body>Test content</body></html>",
                headers={"Content-Type": "text/html"},
            )

            response = await http_client.fetch("https://example.com/page")

            # Should get normal successful response
            assert response.status == 200
            assert b"Test content" in response.body
            assert response.attempts == 1

    @pytest.mark.asyncio
    async def test_robots_txt_error_allows_by_default(self, http_client):
        """Test that robots.txt errors allow crawling by default."""
        with aioresponses() as m:
            # Mock robots.txt returning 404
            m.get("https://example.com/robots.txt", status=404)

            # Mock the actual page
            m.get(
                "https://example.com/page",
                status=200,
                body="<html><body>Test content</body></html>",
                headers={"Content-Type": "text/html"},
            )

            response = await http_client.fetch("https://example.com/page")

            # Should allow crawling when robots.txt not found
            assert response.status == 200
            assert response.body == b"<html><body>Test content</body></html>"
            assert response.attempts == 1

    @pytest.mark.asyncio
    async def test_malformed_meta_robots_allows_by_default(self, http_client):
        """Test that malformed meta robots allows crawling by default."""
        with aioresponses() as m:
            # Mock robots.txt that allows crawling
            m.get("https://example.com/robots.txt", status=200, body="User-agent: *\nAllow: /")

            # Mock page with malformed HTML
            m.get(
                "https://example.com/page",
                status=200,
                body='<html><head><meta name="robots" content=></head><body>Test content</body></html>',
                headers={"Content-Type": "text/html"},
            )

            response = await http_client.fetch("https://example.com/page")

            # Should allow crawling when meta robots can't be parsed
            assert response.status == 200
            assert b"Test content" in response.body
            assert response.attempts == 1

    @pytest.mark.asyncio
    async def test_non_html_content_skips_meta_robots(self, http_client):
        """Test that non-HTML content skips meta robots checking."""
        with aioresponses() as m:
            # Mock robots.txt that allows crawling
            m.get("https://example.com/robots.txt", status=200, body="User-agent: *\nAllow: /")

            # Mock PDF file
            m.get(
                "https://example.com/document.pdf",
                status=200,
                body=b"%PDF-1.4 test pdf content",
                headers={"Content-Type": "application/pdf"},
            )

            response = await http_client.fetch("https://example.com/document.pdf")

            # Should get successful response for non-HTML content
            assert response.status == 200
            assert response.body == b"%PDF-1.4 test pdf content"
            assert response.attempts == 1

    @pytest.mark.asyncio
    async def test_empty_response_allows_crawling(self, http_client):
        """Test that empty responses don't cause issues with meta robots."""
        with aioresponses() as m:
            # Mock robots.txt that allows crawling
            m.get("https://example.com/robots.txt", status=200, body="User-agent: *\nAllow: /")

            # Mock empty response
            m.get("https://example.com/empty", status=200, body="", headers={"Content-Type": "text/html"})

            response = await http_client.fetch("https://example.com/empty")

            # Should handle empty content gracefully
            assert response.status == 200
            assert response.body == b""
            assert response.attempts == 1

    @pytest.mark.asyncio
    async def test_robots_txt_specific_path_restriction(self, robots_enabled_client):
        """Test robots.txt with specific path restrictions."""
        with aioresponses() as m:
            # Mock robots.txt that disallows specific path
            m.get("https://example.com/robots.txt", status=200, body="User-agent: *\nDisallow: /private/\nAllow: /")

            # Test allowed path
            m.get(
                "https://example.com/public/page",
                status=200,
                body="<html><body>Public content</body></html>",
                headers={"Content-Type": "text/html"},
            )

            response = await robots_enabled_client.fetch("https://example.com/public/page")

            # Should allow public path
            assert response.status == 200
            assert response.body == b"<html><body>Public content</body></html>"

            # Test disallowed path
            response = await robots_enabled_client.fetch("https://example.com/private/page")

            # Should block private path
            assert response.status == 999
            assert response.body == b""

    @pytest.mark.asyncio
    async def test_proxy_configuration(self):
        """Test HTTP client with proxy configuration."""
        # Set up proxy environment variable
        os.environ["QUARRY_HTTP_PROXIES"] = "http://proxy1.com:8080,https://proxy2.com:8080"

        try:
            config = Config()
            config.crawler.respect_robots = False
            config.crawler.user_agent = "TestBot/1.0"

            client = HttpClient(config)
            await client.initialize()

            try:
                # Should have loaded proxies
                assert len(client.proxies) == 2
                proxy_hosts = [proxy for proxy in client.proxies]
                assert any("proxy1.com" in proxy for proxy in proxy_hosts)
                assert any("proxy2.com" in proxy for proxy in proxy_hosts)

                # Test proxy rotation
                proxy1 = client._get_next_proxy("https://example.com/page")
                proxy2 = client._get_next_proxy("https://example.com/page")

                # Should get different proxies (or handle appropriately)
                assert proxy1 in client.proxies
                assert proxy2 in client.proxies

            finally:
                await client.close()
        finally:
            # Clean up environment
            os.environ.pop("QUARRY_HTTP_PROXIES", None)

    @pytest.mark.asyncio
    async def test_stats_collection(self, http_client):
        """Test that client statistics are collected properly."""
        # Test initial stats
        stats = http_client.get_stats()

        assert "in_flight_requests" in stats
        assert "domain_semaphores" in stats
        assert "proxy_count" in stats
        assert "backoff_stats" in stats

        assert stats["in_flight_requests"] >= 0
        assert stats["domain_semaphores"] >= 0
        assert stats["proxy_count"] >= 0
        assert isinstance(stats["backoff_stats"], dict)

    @pytest.mark.asyncio
    async def test_domain_semaphore_creation(self, http_client):
        """Test that domain semaphores are created correctly."""
        # Test semaphore creation for different domains
        sem1 = await http_client._get_domain_semaphore("example.com")
        sem2 = await http_client._get_domain_semaphore("test.com")
        sem3 = await http_client._get_domain_semaphore("example.com")  # Same domain

        # Should get different semaphores for different domains
        assert sem1 is not sem2
        # Should get same semaphore for same domain
        assert sem1 is sem3

        # Check stats reflect semaphore creation
        stats = http_client.get_stats()
        assert stats["domain_semaphores"] >= 2

    @pytest.mark.asyncio
    async def test_malformed_url_handling(self, http_client):
        """Test handling of malformed URLs."""
        # Test with invalid URL
        response = await http_client.fetch("not-a-valid-url")

        # Should handle gracefully
        assert response.status == 0  # Error status
        assert response.attempts > 0

    @pytest.mark.asyncio
    async def test_initialization_without_session(self):
        """Test client behavior before initialization."""
        config = Config()
        client = HttpClient(config)

        # Should not have session before initialization
        assert client.session is None
        assert client.robots_cache is None

        await client.initialize()

        # Should have session after initialization
        assert client.session is not None
        assert client.robots_cache is not None

        await client.close()

    @pytest.mark.asyncio
    async def test_double_close_safety(self, http_client):
        """Test that double close doesn't cause issues."""
        # Close once
        await http_client.close()

        # Close again - should not raise exception
        await http_client.close()

        # Session should be None
        assert http_client.session is None
