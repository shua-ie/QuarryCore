"""Performance tests for GPU-accelerated quality scoring."""

from __future__ import annotations

import asyncio
import os
import time
from typing import List

import pytest
import pytest_benchmark
import structlog
from quarrycore.config.config import QualityConfig
from quarrycore.quality.assessor import QualityAssessor
from quarrycore.quality.scorers import TransformerCoherenceScorer

logger = structlog.get_logger(__name__)

# Import torch for skip condition
try:
    import torch
except ImportError:
    torch = None


def generate_test_texts(count: int = 100) -> List[str]:
    """Generate test texts of varying quality and length."""
    texts = []

    # High quality English text
    good_text = """
    Artificial intelligence has revolutionized numerous industries in recent years.
    Machine learning algorithms now power recommendation systems, autonomous vehicles,
    and medical diagnostics. The rapid advancement in deep learning has enabled
    computers to understand and generate human-like text with unprecedented accuracy.
    """

    # Poor quality text
    bad_text = "This bad text. No good grammar here. Very short sentences."

    # Mixed language text
    mixed_text = "This is ein mixed sprache text mit different languages zusammen."

    for i in range(count):
        if i % 3 == 0:
            texts.append(good_text * (i % 5 + 1))  # Varying lengths
        elif i % 3 == 1:
            texts.append(bad_text * (i % 3 + 1))
        else:
            texts.append(mixed_text * (i % 4 + 1))

    return texts


@pytest.mark.benchmark(group="quality-scoring")
class TestQualityPerformance:
    """Performance tests for quality scoring."""

    def test_assessor_latency_cpu(self, benchmark):
        """Benchmark quality assessment latency on CPU."""
        os.environ["QUARRY_TEST_MODE"] = "0"  # Disable test mode for real performance
        try:
            config = QualityConfig(device="cpu", min_score=0.6)
            QualityAssessor._instance = None
            assessor = QualityAssessor(config)

            test_text = generate_test_texts(1)[0]

            # Warm up
            asyncio.run(assessor.score(test_text))

            def score_sync():
                return asyncio.run(assessor.score(test_text))

            result = benchmark(score_sync)

            # Check performance threshold
            assert result is not None
            assert benchmark.stats["mean"] < 0.025  # 25ms threshold

        finally:
            os.environ.pop("QUARRY_TEST_MODE", None)
            if hasattr(assessor, "coherence_scorer"):
                assessor.coherence_scorer.cleanup()
            QualityAssessor._instance = None

    @pytest.mark.skipif(not torch.cuda.is_available() if torch is not None else True, reason="CUDA not available")
    def test_assessor_latency_gpu(self, benchmark):
        """Benchmark quality assessment latency on GPU."""
        os.environ["QUARRY_TEST_MODE"] = "0"  # Disable test mode for real performance
        try:
            config = QualityConfig(device="cuda", min_score=0.6)
            QualityAssessor._instance = None
            assessor = QualityAssessor(config)

            test_text = generate_test_texts(1)[0]

            # Warm up
            asyncio.run(assessor.score(test_text))

            def score_sync():
                return asyncio.run(assessor.score(test_text))

            result = benchmark(score_sync)

            # GPU should be faster
            assert result is not None
            assert benchmark.stats["mean"] < 0.025  # 25ms threshold

        finally:
            os.environ.pop("QUARRY_TEST_MODE", None)
            if hasattr(assessor, "coherence_scorer"):
                assessor.coherence_scorer.cleanup()
            QualityAssessor._instance = None

    def test_batch_throughput(self, benchmark):
        """Test throughput for batch processing."""
        os.environ["QUARRY_TEST_MODE"] = "1"  # Use test mode for consistent results
        try:
            config = QualityConfig(device="cpu", min_score=0.6)
            QualityAssessor._instance = None
            assessor = QualityAssessor(config)

            test_texts = generate_test_texts(10)

            async def score_batch():
                tasks = [assessor.score(text) for text in test_texts]
                return await asyncio.gather(*tasks)

            def score_batch_sync():
                return asyncio.run(score_batch())

            results = benchmark(score_batch_sync)

            # Should process all texts
            assert len(results) == 10
            assert all(0.0 <= score <= 1.0 for score in results)

        finally:
            os.environ.pop("QUARRY_TEST_MODE", None)
            QualityAssessor._instance = None

    def test_memory_usage(self):
        """Test memory usage of quality scoring."""
        import gc

        import psutil

        os.environ["QUARRY_TEST_MODE"] = "1"
        try:
            process = psutil.Process()

            # Get baseline memory
            gc.collect()
            baseline_memory = process.memory_info().rss / 1024 / 1024  # MB

            # Create assessor
            config = QualityConfig(device="cpu", min_score=0.6)
            QualityAssessor._instance = None
            assessor = QualityAssessor(config)

            # Process many texts
            test_texts = generate_test_texts(100)

            async def process_all():
                for text in test_texts:
                    await assessor.score(text)

            asyncio.run(process_all())

            # Check memory after processing
            gc.collect()
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_delta = final_memory - baseline_memory

            # Should not use more than 300MB
            assert memory_delta < 300, f"Memory usage too high: {memory_delta}MB"

        finally:
            os.environ.pop("QUARRY_TEST_MODE", None)
            if "assessor" in locals() and hasattr(assessor, "cleanup"):
                assessor.cleanup()
            QualityAssessor._instance = None

    def test_concurrent_scoring(self, benchmark):
        """Test concurrent scoring performance."""
        os.environ["QUARRY_TEST_MODE"] = "1"
        try:
            config = QualityConfig(device="cpu", min_score=0.6)
            QualityAssessor._instance = None
            assessor = QualityAssessor(config)

            test_texts = generate_test_texts(20)

            async def concurrent_score():
                # Simulate concurrent requests
                tasks = []
                for text in test_texts:
                    tasks.append(assessor.score(text))
                return await asyncio.gather(*tasks)

            def run_concurrent():
                return asyncio.run(concurrent_score())

            results = benchmark(run_concurrent)

            # All should complete successfully
            assert len(results) == 20
            assert all(isinstance(score, float) for score in results)

        finally:
            os.environ.pop("QUARRY_TEST_MODE", None)
            QualityAssessor._instance = None
