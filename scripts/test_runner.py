#!/usr/bin/env python3
"""
Simple test runner to show pytest progress in real-time
"""
import subprocess
import sys
from pathlib import Path


def main():
    print("🚀 Running pytest with real-time output...")
    print("=" * 60)

    # Run pytest with coverage, showing output in real-time
    cmd = [
        "python",
        "-m",
        "pytest",
        "--cov=src/quarrycore",
        "--cov-branch",
        "--cov-report=xml:coverage.xml",
        "--cov-report=term-missing",
        "-v",
        "--tb=short",
    ]

    print(f"Command: {' '.join(cmd)}")
    print("=" * 60)

    try:
        # Run with real-time output
        result = subprocess.run(cmd, cwd=Path(__file__).parent.parent, text=True)

        print("=" * 60)
        print(f"Exit code: {result.returncode}")

        return result.returncode

    except Exception as e:
        print(f"Error running tests: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
