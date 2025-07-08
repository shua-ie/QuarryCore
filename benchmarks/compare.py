#!/usr/bin/env python3
"""
Compare benchmark results against baseline performance.

Usage: python benchmarks/compare.py [threshold_percentage]

Exits with code 0 if performance retention is >= threshold, 1 otherwise.
"""

import json
import sys
from pathlib import Path


def load_json(filepath: Path) -> dict:
    """Load JSON data from file."""
    with open(filepath, "r") as f:
        return json.load(f)


def calculate_retention(baseline: dict, current: dict) -> float:
    """
    Calculate performance retention percentage.

    Higher throughput is better, so retention = (current / baseline) * 100
    """
    # Handle different data structures
    # baseline.json uses performance_metrics.urls_per_second
    # benchmark_results.json uses throughput_urls_per_second

    if "performance_metrics" in baseline:
        baseline_throughput = baseline["performance_metrics"].get("urls_per_second", 0)
    else:
        baseline_throughput = baseline.get("throughput_urls_per_second", 0)

    current_throughput = current.get("throughput_urls_per_second", 0)

    if baseline_throughput == 0:
        return 100.0

    retention = (current_throughput / baseline_throughput) * 100
    return retention


def main():
    """Main comparison function."""
    # Parse arguments
    threshold = 95.0  # Default threshold
    benchmarks_dir = None

    if len(sys.argv) > 1:
        try:
            threshold = float(sys.argv[1])
        except ValueError:
            print(f"Error: Invalid threshold value '{sys.argv[1]}'", file=sys.stderr)
            sys.exit(1)

    # Optional: Accept directory path as second argument (for testing)
    if len(sys.argv) > 2:
        benchmarks_dir = Path(sys.argv[2])
    else:
        benchmarks_dir = Path(__file__).parent

    # Paths
    baseline_file = benchmarks_dir / "baseline.json"
    results_file = benchmarks_dir / "benchmark_results.json"

    # Check if files exist
    if not baseline_file.exists():
        print(f"Error: Baseline file not found: {baseline_file}", file=sys.stderr)
        sys.exit(1)

    if not results_file.exists():
        print(f"Error: Results file not found: {results_file}", file=sys.stderr)
        print("Please run 'python benchmarks/profile_1k_urls.py' first", file=sys.stderr)
        sys.exit(1)

    # Load data
    baseline = load_json(baseline_file)
    current = load_json(results_file)

    # Calculate retention
    retention = calculate_retention(baseline, current)

    # Print comparison
    print("=== Benchmark Comparison Report ===")

    # Handle different data structures for baseline
    if "performance_metrics" in baseline:
        baseline_throughput = baseline["performance_metrics"].get("urls_per_second", 0)
    else:
        baseline_throughput = baseline.get("throughput_urls_per_second", 0)

    current_throughput = current.get("throughput_urls_per_second", 0)

    print(f"Baseline throughput: {baseline_throughput:.2f} URLs/sec")
    print(f"Current throughput:  {current_throughput:.2f} URLs/sec")
    print(f"Performance retention: {retention:.1f}%")
    print(f"Required threshold: {threshold:.1f}%")

    # Detailed comparison if available
    if "stage_stats" in baseline and "stage_stats" in current:
        print("\nStage-by-stage comparison:")
        baseline_stages = baseline["stage_stats"]
        current_stages = current.get("stage_stats", {})

        for stage in baseline_stages:
            if stage in current_stages:
                baseline_avg = baseline_stages[stage]["avg_duration"]
                current_avg = current_stages[stage].get("avg_duration", 0)
                if baseline_avg > 0:
                    stage_retention = (baseline_avg / current_avg) * 100 if current_avg > 0 else 0
                    print(
                        f"  {stage}: {stage_retention:.1f}% retention "
                        f"({current_avg:.3f}s vs {baseline_avg:.3f}s baseline)"
                    )

    print("=" * 35)

    # Exit based on threshold
    if retention >= threshold:
        print(f"\n✅ BENCHMARK PASSED: Throughput retention {retention:.1f}% ≥ {threshold:.1f}% - OK")
        print("Performance requirements met successfully.")
        sys.exit(0)
    else:
        print(
            f"\n❌ BENCHMARK FAILED: Throughput retention {retention:.1f}% < {threshold:.1f}% - NOT OK", file=sys.stderr
        )
        print(
            f"Performance degradation detected! Current throughput is only {retention:.1f}% of baseline.",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
