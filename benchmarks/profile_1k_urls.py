#!/usr/bin/env python3
"""
Benchmark script for profiling pipeline performance with 1000 URLs.

Uses httpx mocking to mock HTTP calls for reproducible benchmarking.
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Dict, List
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from quarrycore.container import DependencyContainer
from quarrycore.pipeline import Pipeline, PipelineSettings
from quarrycore.quality.quality_assessor import QualityScore


async def generate_mock_urls(count: int = 1000) -> List[str]:
    """Generate test URLs for benchmarking."""
    urls = []
    for i in range(count):
        domain_idx = i % 10  # Use 10 different domains
        urls.append(f"https://example{domain_idx}.com/page{i}")
    return urls


def create_mock_response(url: str, index: int) -> AsyncMock:
    """Create a mock HTTP response for a URL."""
    # Generate simple HTML content
    html_content = f"""
    <html>
    <head><title>Page {index} - Test Content</title></head>
    <body>
        <h1>Page {index}</h1>
        <p>This is test content for benchmarking purposes.
        Lorem ipsum dolor sit amet, consectetur adipiscing elit.
        Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.</p>
        <p>More content to make it realistic. The quick brown fox jumps over the lazy dog.
        Pack my box with five dozen liquor jugs. How vexingly quick daft zebras jump!</p>
    </body>
    </html>
    """

    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.url = url
    mock_response.content = html_content.encode("utf-8")
    mock_response.headers = {"content-type": "text/html; charset=utf-8", "content-length": str(len(html_content))}
    return mock_response


async def run_benchmark() -> Dict[str, float]:
    """Run the benchmark and return performance metrics."""
    # Generate URLs
    urls = await generate_mock_urls(1000)

    # Configure pipeline for benchmarking
    settings = PipelineSettings(
        checkpoint_interval=300.0,  # Less frequent checkpoints during benchmark
        domain_failure_threshold=10,
        domain_backoff_duration=30.0,
    )

    container = DependencyContainer()
    pipeline = Pipeline(container, max_concurrency=50, settings=settings)

    # Track timing
    start_time = time.time()

    # Mock both httpx and aiohttp to avoid real network I/O and
    # ensure the production HttpClient used inside the Pipeline
    # returns successful responses quickly. This keeps the benchmark
    # representative while making it deterministic and fast.

    # Helper to build an aiohttp-like mock response object.
    class _AioHTTPResponseMock(AsyncMock):
        def __init__(self, url: str):
            super().__init__()
            html = "<html><body>Mock</body></html>"
            self.status = 200
            self.url = url
            self._body = html.encode()
            self.headers = {
                "content-type": "text/html; charset=utf-8",
                "content-length": str(len(html)),
            }

        async def text(self):
            return self._body.decode()

        async def read(self):
            return self._body

    async def _aiohttp_get_side_effect(url, *args, **kwargs):
        return _AioHTTPResponseMock(url)

    from quarrycore.crawler.http_client import CrawlerResponse  # Local import to avoid circular deps

    async def _mock_fetch(url: str, *_, **__) -> CrawlerResponse:  # type: ignore[override]
        return CrawlerResponse(
            status=200,
            headers={"content-type": "text/html"},
            body=b"<html>OK</html>",
            start_ts=time.time(),
            end_ts=time.time(),
            attempts=1,
            url=url,
            final_url=url,
        )

    with (
        # Mock httpx (used by some components)
        patch("httpx.AsyncClient") as mock_httpx_client,
        # Mock HttpClient.fetch to bypass network entirely
        patch("quarrycore.crawler.http_client.HttpClient.fetch", side_effect=_mock_fetch),
        # Mock QualityAssessor to return perfect score instantly
        patch(
            "quarrycore.quality.quality_assessor.QualityAssessor.assess_quality",
            new=AsyncMock(return_value=QualityScore(overall_score=1.0)),
        ),
        # Mock StorageManager.store_extracted_content to just return a UUID without IO
        patch(
            "quarrycore.storage.storage_manager.StorageManager.store_extracted_content",
            new=AsyncMock(return_value=uuid4()),
        ),
    ):
        # httpx client instance setup (mirrors previous behaviour)
        mock_httpx_instance = AsyncMock()
        mock_httpx_client.return_value.__aenter__.return_value = mock_httpx_instance

        def httpx_get_side_effect(url, **_):
            idx = int(url.split("page")[-1])
            return create_mock_response(url, idx)

        mock_httpx_instance.get.side_effect = httpx_get_side_effect

        # Run the pipeline – now entirely mocked underneath
        result = await pipeline.run(urls=urls, batch_size=100, checkpoint_interval=300.0)

    end_time = time.time()
    duration = end_time - start_time

    # Calculate metrics
    throughput = result["processed_count"] / duration  # URLs per second

    metrics = {
        "duration_seconds": duration,
        "total_urls": len(urls),
        "processed_count": result["processed_count"],
        "failed_count": result["failed_count"],
        "throughput_urls_per_second": throughput,
        "avg_time_per_url_ms": (duration / result["processed_count"]) * 1000 if result["processed_count"] > 0 else 0,
    }

    # Get stage timings if available
    if hasattr(pipeline, "get_performance_stats"):
        stage_stats = pipeline.get_performance_stats()
        metrics["stage_stats"] = stage_stats

    return metrics


async def main():
    """Run benchmark and save results."""
    print("Running pipeline benchmark with 1000 URLs...")

    # Run benchmark
    metrics = await run_benchmark()

    # Print results
    print("\nBenchmark Results:")
    print(f"  Duration: {metrics['duration_seconds']:.2f} seconds")
    print(f"  Throughput: {metrics['throughput_urls_per_second']:.2f} URLs/second")
    print(f"  Avg time per URL: {metrics['avg_time_per_url_ms']:.2f} ms")
    print(f"  Processed: {metrics['processed_count']}/{metrics['total_urls']}")
    print(f"  Failed: {metrics['failed_count']}")

    # Save results
    results_file = Path(__file__).parent / "benchmark_results.json"
    with open(results_file, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nResults saved to: {results_file}")

    return metrics


if __name__ == "__main__":
    # Set test mode to use mock models
    import os

    os.environ["QUARRY_TEST_MODE"] = "1"

    asyncio.run(main())
