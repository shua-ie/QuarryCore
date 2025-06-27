"""
Unit tests for benchmark comparison script.

BENCH-02: Verify script exits non-zero when retention < 95%.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


def test_benchmark_compare_success():
    """Test benchmark compare passes when retention >= threshold."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create baseline with good performance
        baseline = {"throughput_urls_per_second": 50.0, "duration_seconds": 20.0, "processed_count": 1000}
        baseline_file = temp_path / "baseline.json"
        with open(baseline_file, "w") as f:
            json.dump(baseline, f)

        # Create results with similar performance (100% retention)
        results = {"throughput_urls_per_second": 50.0, "duration_seconds": 20.0, "processed_count": 1000}
        results_file = temp_path / "benchmark_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f)

        # Run compare script
        cmd = [sys.executable, str(Path.cwd() / "benchmarks" / "compare.py"), "95", str(temp_path)]
        proc = subprocess.run(cmd, capture_output=True, text=True, env={"PYTHONPATH": str(Path.cwd())})

        # Should exit 0
        assert proc.returncode == 0, f"Expected exit 0, got {proc.returncode}\nStderr: {proc.stderr}"

        # Should print success message
        assert "BENCHMARK PASSED" in proc.stdout
        assert "≥ 95.0% - OK" in proc.stdout


def test_benchmark_compare_failure():
    """
    BENCH-02: Test benchmark compare fails when retention < threshold.

    Expects SystemExit with code != 0.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create baseline with good performance
        baseline = {"throughput_urls_per_second": 50.0, "duration_seconds": 20.0, "processed_count": 1000}
        baseline_file = temp_path / "baseline.json"
        with open(baseline_file, "w") as f:
            json.dump(baseline, f)

        # Create results with poor performance (60% retention)
        results = {
            "throughput_urls_per_second": 30.0,  # Only 60% of baseline
            "duration_seconds": 33.3,
            "processed_count": 1000,
        }
        results_file = temp_path / "benchmark_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f)

        # Run compare script
        cmd = [sys.executable, str(Path.cwd() / "benchmarks" / "compare.py"), "95", str(temp_path)]
        proc = subprocess.run(cmd, capture_output=True, text=True, env={"PYTHONPATH": str(Path.cwd())})

        # Should exit non-zero
        assert proc.returncode != 0, f"Expected non-zero exit, got {proc.returncode}"

        # Should print failure message
        assert "BENCHMARK FAILED" in proc.stderr
        assert "< 95.0% - NOT OK" in proc.stderr
        assert "60.0%" in proc.stdout or "60.0%" in proc.stderr


def test_benchmark_compare_missing_files():
    """Test benchmark compare handles missing files gracefully."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Run from empty temp directory
        cmd = [sys.executable, str(Path.cwd() / "benchmarks" / "compare.py"), "95", temp_dir]
        proc = subprocess.run(cmd, capture_output=True, text=True)

        # Should exit with error
        assert proc.returncode == 1
        assert "not found" in proc.stderr


def test_benchmark_compare_invalid_threshold():
    """Test benchmark compare handles invalid threshold argument."""
    benchmarks_dir = Path.cwd() / "benchmarks"
    cmd = [sys.executable, str(benchmarks_dir / "compare.py"), "invalid"]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=benchmarks_dir)

    # Should exit with error
    assert proc.returncode == 1
    assert "Invalid threshold" in proc.stderr


def test_benchmark_compare_edge_cases():
    """Test edge cases like zero baseline throughput."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create baseline with zero throughput
        baseline = {"throughput_urls_per_second": 0.0, "duration_seconds": 0, "processed_count": 0}
        baseline_file = temp_path / "baseline.json"
        with open(baseline_file, "w") as f:
            json.dump(baseline, f)

        # Create results
        results = {"throughput_urls_per_second": 50.0, "duration_seconds": 20.0, "processed_count": 1000}
        results_file = temp_path / "benchmark_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f)

        # Run compare script
        cmd = [sys.executable, str(Path.cwd() / "benchmarks" / "compare.py"), "95", str(temp_path)]
        proc = subprocess.run(cmd, capture_output=True, text=True, env={"PYTHONPATH": str(Path.cwd())})

        # Should handle gracefully (100% retention when baseline is 0)
        assert proc.returncode == 0


def test_benchmark_failure_expectation():
    """
    BENCH-02 verification: Test that expects benchmark failure.

    This test verifies sys.exit with non-zero code when performance degrades.
    """
    # This simulates what would happen in CI if performance degrades
    with pytest.raises(SystemExit) as exc_info:
        # Simulate running the compare script with poor performance
        import benchmarks.compare

        # Mock the file loading to return poor performance
        def mock_load_json(filepath):
            if "baseline" in str(filepath):
                return {"throughput_urls_per_second": 100.0}
            else:
                return {"throughput_urls_per_second": 50.0}  # 50% of baseline

        # Patch and run
        original_load_json = benchmarks.compare.load_json
        try:
            benchmarks.compare.load_json = mock_load_json
            sys.argv = ["compare.py", "95"]
            benchmarks.compare.main()
        finally:
            benchmarks.compare.load_json = original_load_json

    # Should exit with non-zero code
    assert exc_info.value.code != 0
