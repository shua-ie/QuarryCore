"""Tests for the crawler module."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch

from quarrycore.crawler.adaptive_crawler import AdaptiveConfig
from quarrycore.crawler import AdaptiveCrawler


class TestCrawlerConfig:
    """Tests for CrawlerConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = AdaptiveConfig()
        assert config.max_concurrent_requests > 0
        assert config.request_delay_seconds >= 0
        assert config.max_retries >= 0


class TestAdaptiveCrawler:
    """Tests for AdaptiveCrawler."""
    
    @pytest.fixture
    def crawler_config(self):
        """Provide test crawler configuration."""
        return AdaptiveConfig(
            max_concurrent_requests=5,
            request_delay_seconds=0,
            max_retries=2,
        )
    
    def test_crawler_initialization(self, crawler_config):
        """Test crawler initialization."""
        crawler = AdaptiveCrawler(adaptive_config=crawler_config)
        assert crawler.adaptive_config == crawler_config
    
    @pytest.mark.asyncio
    async def test_crawl_single_url(self, crawler_config, mock_http_client):
        """Test crawling a single URL."""
        with patch('httpx.AsyncClient', return_value=mock_http_client):
            crawler = AdaptiveCrawler(adaptive_config=crawler_config)
            # Add specific test implementation
            pass
    
    @pytest.mark.asyncio 
    async def test_crawl_multiple_urls(self, crawler_config):
        """Test crawling multiple URLs concurrently."""
        crawler = AdaptiveCrawler(adaptive_config=crawler_config)
        urls = ["https://example.com/1", "https://example.com/2"]
        # Add specific test implementation
        pass 