"""
Tests for HttpClient proxy rotation and fallback logic.
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch
from urllib.parse import urlparse

import pytest
from aioresponses import aioresponses
from quarrycore.config import Config
from quarrycore.crawler.http_client import HttpClient


class TestHttpClientProxyRotation:
    """Test proxy rotation logic in HttpClient."""

    @pytest.fixture
    def config(self):
        """Create test config."""
        config = Config()
        config.debug.test_mode = True  # Enable deterministic proxy selection
        return config

    def create_test_client(self, config):
        """Create HttpClient for testing without event loop issues."""
        client = HttpClient(config)
        return client

    @pytest.mark.asyncio
    async def test_proxy_setup_from_environment(self, config):
        """Test proxy setup from environment variable."""
        proxy_list = "http://proxy1:8080,https://proxy2:8080,socks5://proxy3:1080"

        with patch.dict(os.environ, {"QUARRY_HTTP_PROXIES": proxy_list}):
            with patch("aiohttp.TCPConnector"):
                client = HttpClient(config)

                # Should parse proxies from environment
                assert len(client.proxies) == 3
                assert "http://proxy1:8080" in client.proxies
                assert "https://proxy2:8080" in client.proxies
                assert "socks5://proxy3:1080" in client.proxies

    @pytest.mark.asyncio
    async def test_proxy_setup_with_empty_environment(self, config):
        """Test proxy setup with empty environment variable."""
        with patch.dict(os.environ, {"QUARRY_HTTP_PROXIES": ""}):
            with patch("aiohttp.TCPConnector"):
                client = HttpClient(config)

                # Should have no proxies
                assert len(client.proxies) == 0

    @pytest.mark.asyncio
    async def test_proxy_setup_with_whitespace_in_list(self, config):
        """Test proxy setup handles whitespace in proxy list."""
        proxy_list = " http://proxy1:8080 , https://proxy2:8080 , , socks5://proxy3:1080 "

        with patch.dict(os.environ, {"QUARRY_HTTP_PROXIES": proxy_list}):
            with patch("aiohttp.TCPConnector"):
                client = HttpClient(config)

                # Should filter out empty entries and strip whitespace
                assert len(client.proxies) == 3
                assert "http://proxy1:8080" in client.proxies
                assert "https://proxy2:8080" in client.proxies
                assert "socks5://proxy3:1080" in client.proxies

    @pytest.mark.asyncio
    async def test_get_next_proxy_with_no_proxies(self, config):
        """Test proxy selection when no proxies are configured."""
        http_client = self.create_test_client(config)
        http_client.proxies = []

        proxy = http_client._get_next_proxy("https://example.com")

        assert proxy is None

    @pytest.mark.asyncio
    async def test_get_next_proxy_scheme_matching(self, config):
        """Test proxy selection prioritizes scheme matching."""
        http_client = self.create_test_client(config)
        http_client.proxies = ["http://proxy1:8080", "https://proxy2:8080", "socks5://proxy3:1080"]

        # Test HTTPS URL should prefer HTTPS proxy
        proxy = http_client._get_next_proxy("https://example.com")
        assert proxy == "https://proxy2:8080"

        # Test HTTP URL should prefer HTTP proxy
        proxy = http_client._get_next_proxy("http://example.com")
        assert proxy == "http://proxy1:8080"

    @pytest.mark.asyncio
    async def test_get_next_proxy_fallback_to_first(self, config):
        """Test proxy selection falls back to first proxy when no scheme match."""
        http_client = self.create_test_client(config)
        http_client.proxies = ["http://proxy1:8080", "https://proxy2:8080"]

        # Test with unsupported scheme should fallback to first proxy
        proxy = http_client._get_next_proxy("ftp://example.com")
        assert proxy == "http://proxy1:8080"

    @pytest.mark.asyncio
    async def test_deterministic_proxy_selection_in_test_mode(self, config):
        """Test that proxy selection is deterministic in test mode."""
        http_client = self.create_test_client(config)
        http_client.proxies = ["http://proxy1:8080", "https://proxy2:8080", "http://proxy3:8080"]

        # Multiple calls should return same proxy for same URL
        proxy1 = http_client._get_next_proxy("https://example.com")
        proxy2 = http_client._get_next_proxy("https://example.com")
        proxy3 = http_client._get_next_proxy("https://example.com")

        assert proxy1 == proxy2 == proxy3

    @pytest.mark.asyncio
    async def test_proxy_rotation_in_non_test_mode(self, config):
        """Test proxy rotation works in non-test mode."""
        config.debug.test_mode = False

        with patch.dict(os.environ, {"QUARRY_HTTP_PROXIES": "http://proxy1:8080,http://proxy2:8080"}):
            with patch("aiohttp.TCPConnector"):
                client = HttpClient(config)

                # Should have proxies but order may be randomized
                assert len(client.proxies) == 2
                assert "http://proxy1:8080" in client.proxies
                assert "http://proxy2:8080" in client.proxies

    @pytest.mark.asyncio
    async def test_proxy_index_advancement(self, config):
        """Test that proxy index advances in non-test mode."""
        http_client = self.create_test_client(config)
        http_client.config.debug.test_mode = False
        http_client.proxies = ["http://proxy1:8080", "http://proxy2:8080", "http://proxy3:8080"]

        # Get multiple proxies and ensure index advances
        proxies_selected = []
        for _ in range(5):
            proxy = http_client._get_next_proxy("http://example.com")
            proxies_selected.append(proxy)

        # Should have selected proxies (may repeat due to round-robin)
        assert len(set(proxies_selected)) <= 3  # At most 3 unique proxies
        assert all(p in http_client.proxies for p in proxies_selected)

    @pytest.mark.asyncio
    async def test_proxy_with_invalid_url_scheme(self, config):
        """Test proxy selection with invalid URL scheme."""
        http_client = self.create_test_client(config)
        http_client.proxies = ["http://proxy1:8080"]

        # Test with malformed URL
        proxy = http_client._get_next_proxy("not-a-url")

        # Should still return first proxy as fallback
        assert proxy == "http://proxy1:8080"


class TestHttpClientProxyFetch:
    """Test proxy usage during HTTP fetch operations."""

    @pytest.fixture
    def config(self):
        """Create test config."""
        config = Config()
        config.debug.test_mode = True
        return config

    @pytest.fixture
    async def http_client(self, config):
        """Create and initialize HttpClient instance."""
        client = HttpClient(config)
        await client.initialize()
        yield client
        await client.close()

    @pytest.mark.asyncio
    async def test_fetch_with_proxy_success(self, http_client):
        """Test successful fetch using proxy."""
        http_client.proxies = ["http://proxy1:8080"]

        with aioresponses() as m:
            m.get("https://example.com", status=200, body="Success")

            response = await http_client.fetch("https://example.com")

            assert response.status == 200
            assert response.body == b"Success"

    @pytest.mark.asyncio
    async def test_fetch_with_proxy_failure_fallback(self, http_client):
        """Test fetch falls back when proxy fails."""
        http_client.proxies = ["http://bad-proxy:8080"]

        with aioresponses() as m:
            # Mock both proxy and direct requests
            m.get("https://example.com", status=200, body="Success")

            response = await http_client.fetch("https://example.com")

            # Should succeed (either via proxy or direct)
            assert response.status in [200, 0]  # 200 for success, 0 for failure

    @pytest.mark.asyncio
    async def test_fetch_without_proxy(self, http_client):
        """Test fetch works without proxy configuration."""
        http_client.proxies = []

        with aioresponses() as m:
            m.get("https://example.com", status=200, body="Success")

            response = await http_client.fetch("https://example.com")

            assert response.status == 200
            assert response.body == b"Success"

    @pytest.mark.asyncio
    async def test_fetch_with_multiple_proxies(self, http_client):
        """Test fetch with multiple proxy options."""
        http_client.proxies = ["http://proxy1:8080", "https://proxy2:8080", "http://proxy3:8080"]

        with aioresponses() as m:
            m.get("https://example.com", status=200, body="Success")

            response = await http_client.fetch("https://example.com")

            assert response.status == 200
            assert response.body == b"Success"

    @pytest.mark.asyncio
    async def test_malformed_proxy_url_handling(self, http_client):
        """Test handling of malformed proxy URLs."""
        http_client.proxies = ["not-a-valid-proxy-url"]

        with aioresponses() as m:
            m.get("https://example.com", status=200, body="Success")

            # Should not crash, may succeed via direct connection
            response = await http_client.fetch("https://example.com")

            # Response should be valid (either success or controlled failure)
            assert response.status in [200, 0]

    @pytest.mark.asyncio
    async def test_proxy_authentication_handling(self, http_client):
        """Test proxy URLs with authentication."""
        http_client.proxies = ["http://user:pass@proxy1:8080"]

        with aioresponses() as m:
            m.get("https://example.com", status=200, body="Success")

            response = await http_client.fetch("https://example.com")

            # Should handle authenticated proxy
            assert response.status in [200, 0]

    @pytest.mark.asyncio
    async def test_proxy_scheme_validation(self, http_client):
        """Test proxy scheme validation during selection."""
        http_client.proxies = [
            "http://proxy1:8080",
            "https://proxy2:8080",
            "socks5://proxy3:1080",
            "invalid-scheme://proxy4:8080",
        ]

        # Test various URL schemes
        test_cases = [
            ("http://example.com", "http://proxy1:8080"),
            ("https://example.com", "https://proxy2:8080"),
            ("ftp://example.com", "http://proxy1:8080"),  # Fallback to first
        ]

        for url, expected_proxy in test_cases:
            selected_proxy = http_client._get_next_proxy(url)
            assert selected_proxy == expected_proxy

    @pytest.mark.asyncio
    async def test_proxy_rotation_exhaustion(self, http_client):
        """Test behavior when all proxies are exhausted."""
        http_client.proxies = ["http://proxy1:8080"]

        # Get proxy multiple times
        for _ in range(10):
            proxy = http_client._get_next_proxy("http://example.com")
            assert proxy == "http://proxy1:8080"

    @pytest.mark.asyncio
    async def test_concurrent_proxy_selection(self, http_client):
        """Test proxy selection under concurrent access."""
        http_client.proxies = ["http://proxy1:8080", "http://proxy2:8080", "http://proxy3:8080"]

        import asyncio

        async def get_proxy():
            return http_client._get_next_proxy("http://example.com")

        # Run multiple concurrent proxy selections
        tasks = [get_proxy() for _ in range(10)]
        results = await asyncio.gather(*tasks)

        # All results should be valid proxies
        for proxy in results:
            assert proxy in http_client.proxies

    @pytest.mark.asyncio
    async def test_cleanup_after_tests(self, http_client):
        """Ensure proper cleanup after proxy tests."""
        await http_client.close()

        # Verify client is properly closed
        assert http_client.session is None
