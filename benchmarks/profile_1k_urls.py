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

from quarrycore.container import DependencyContainer
from quarrycore.pipeline import Pipeline, PipelineSettings


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

    # Run with mocked responses
    with patch("httpx.AsyncClient") as mock_client:
        # Create mock client instance
        mock_client_instance = AsyncMock()
        mock_client.return_value.__aenter__.return_value = mock_client_instance

        # Set up all mock responses
        def get_side_effect(url, **kwargs):
            # Extract index from URL
            index = int(url.split("page")[-1])
            return create_mock_response(url, index)

        mock_client_instance.get.side_effect = get_side_effect

        # Run pipeline
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
