#!/usr/bin/env python3
"""
Smoke test for QuarryCore - processes 20 real URLs to verify system health.
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from quarrycore.container import DependencyContainer  # noqa: E402
from quarrycore.pipeline import Pipeline  # noqa: E402


async def run_smoke_test(count: int = 20) -> dict:
    """Run smoke test with specified number of URLs."""
    # Use a mix of real URLs for testing
    test_urls = [
        "https://example.com",
        "https://www.python.org",
        "https://docs.python.org/3/",
        "https://pypi.org",
        "https://github.com",
        "https://stackoverflow.com",
        "https://www.wikipedia.org",
        "https://news.ycombinator.com",
        "https://www.reddit.com",
        "https://www.bbc.com",
        "https://www.nytimes.com",
        "https://www.theguardian.com",
        "https://www.reuters.com",
        "https://www.bloomberg.com",
        "https://www.techcrunch.com",
        "https://www.wired.com",
        "https://www.nature.com",
        "https://www.sciencedirect.com",
        "https://arxiv.org",
        "https://www.jstor.org",
    ][:count]

    container = DependencyContainer()
    pipeline = Pipeline(container)

    print(f"Starting smoke test with {count} URLs...")

    try:
        result = await pipeline.run(test_urls)

        # Check results
        processed = result.get("processed_count", 0)
        failed = result.get("failed_count", 0)
        total = processed + failed

        print("\nResults:")
        print(f"  Total URLs: {total}")
        print(f"  Processed: {processed}")
        print(f"  Failed: {failed}")
        print(f"  Success rate: {processed/total*100:.1f}%")

        # Check if we have non-None content for at least 90% of URLs (18/20)
        success_threshold = int(count * 0.9)
        if processed >= success_threshold:
            print(f"\n✅ PASSED: {processed}/{count} URLs processed successfully (>= {success_threshold} required)")
            return {"status": "PASSED", "processed": processed, "failed": failed, "total": total}
        else:
            print(f"\n❌ FAILED: Only {processed}/{count} URLs processed successfully (>= {success_threshold} required)")
            return {"status": "FAILED", "processed": processed, "failed": failed, "total": total}

    except Exception as e:
        print(f"\n❌ FAILED: Uncaught exception: {e}")
        return {"status": "FAILED", "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="QuarryCore smoke test")
    parser.add_argument("--count", type=int, default=20, help="Number of URLs to test")
    args = parser.parse_args()

    result = asyncio.run(run_smoke_test(args.count))

    if result["status"] == "FAILED":
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
