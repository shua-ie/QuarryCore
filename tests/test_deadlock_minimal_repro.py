"""
Minimal test to reproduce and diagnose the async deadlock issue.
"""

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from quarrycore.crawler.adaptive_crawler import AdaptiveCrawler

# Set up logging to see diagnostic output
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


@pytest.mark.asyncio
async def test_minimal_deadlock_reproduction():
    """Minimal test to reproduce the deadlock with full diagnostic logging."""

    # Create mock client
    mock_client = AsyncMock()

    async def mock_get(*args, **kwargs):
        await asyncio.sleep(0.01)  # Small delay
        return MagicMock(
            status_code=200,
            headers={"content-type": "text/html"},
            content=b"<html>test</html>",
            text="<html>test</html>",
            url=args[0] if args else "https://example.com",
        )

    mock_client.get = mock_get

    with patch("httpx.AsyncClient", return_value=mock_client):
        # Create simple test case
        urls = [f"https://example.com/test-{i}" for i in range(5)]

        crawler = AdaptiveCrawler()

        async with crawler:
            results = []
            async for result in crawler.crawl_batch(urls, concurrency=2):
                results.append(result)
                if len(results) >= 3:  # Stop early
                    break

        assert len(results) > 0, "Should have processed some URLs"


@pytest.mark.asyncio
async def test_performance_pattern_reproduction():
    """Reproduce the specific pattern from performance tests that causes deadlock."""

    # Hardware caps like performance tests
    hardware_caps = MagicMock()
    hardware_caps.max_concurrency = 50
    hardware_caps.max_memory_gb = 16
    hardware_caps.cpu_cores = 8
    hardware_caps.memory_gb = 16
    hardware_caps.has_gpu = False
    hardware_caps.available_memory_gb = 12.0  # Add this missing attribute

    # Mock client like performance tests
    mock_client = AsyncMock()

    async def perf_mock_get(*args, **kwargs):
        await asyncio.sleep(0.01)  # 10ms like performance tests
        return MagicMock(
            status_code=200,
            headers={"content-type": "text/html"},
            content=b"<html><body>Mock content for testing</body></html>",
            text="<html><body>Mock content for testing</body></html>",
            url=args[0] if args else "https://example.com",
        )

    mock_client.get = perf_mock_get

    with patch("httpx.AsyncClient", return_value=mock_client):
        crawler = AdaptiveCrawler(hardware_caps=hardware_caps)

        async with crawler:
            # Test multiple batches like sustained load test
            for batch_num in range(3):  # Limited batches
                batch_urls = [f"https://example.com/batch-{batch_num}-{i}" for i in range(10)]

                results = []
                async for result in crawler.crawl_batch(batch_urls, concurrency=5):
                    results.append(result)
                    if len(results) >= 5:  # Stop early
                        break

                # Brief pause between batches like real test
                await asyncio.sleep(0.01)

        assert True, "Test completed successfully"
