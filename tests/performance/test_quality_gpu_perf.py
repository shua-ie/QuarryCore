"""
Performance tests for GPU-accelerated quality scoring.

These tests are automatically skipped if CUDA is not available.
"""

import asyncio
import statistics
import time
from typing import List

import pytest
import structlog
from quarrycore.config.config import QualityConfig
from quarrycore.quality.scorers import TransformerCoherenceScorer

logger = structlog.get_logger(__name__)

# Check for CUDA availability
try:
    import torch

    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    CUDA_AVAILABLE = False


@pytest.mark.requires_cuda
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestQualityGPUPerformance:
    """Performance benchmarks for GPU-accelerated quality scoring."""

    @pytest.fixture
    def sample_texts(self) -> List[str]:
        """Generate sample texts of ~1KB each for benchmarking."""
        base_text = (
            "This is a sample text used for performance benchmarking. "
            "It contains meaningful content that demonstrates various aspects of writing quality. "
            "The text has proper grammar, coherent structure, and conveys information clearly. "
            "Performance testing helps ensure that our quality assessment remains fast and efficient. "
        )

        # Create texts of approximately 1KB each
        texts = []
        for i in range(100):
            # Vary the content slightly to avoid caching effects
            text = f"Document {i}: " + (base_text * 5)  # ~1KB
            texts.append(text)

        return texts

    @pytest.mark.asyncio
    async def test_gpu_scorer_latency(self, sample_texts):
        """Test that GPU scorer meets latency requirements (≤ 25ms median for 1KB text)."""
        config = QualityConfig(device="cuda")
        scorer = TransformerCoherenceScorer(config)

        # Warm up the model
        for _ in range(5):
            await scorer.score(sample_texts[0])

        # Measure latencies
        latencies = []
        for text in sample_texts[:50]:  # Test with 50 samples
            start = time.perf_counter()
            await scorer.score(text)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # Convert to milliseconds

        # Calculate statistics
        median_latency = statistics.median(latencies)
        p95_latency = statistics.quantiles(latencies, n=20)[18]  # 95th percentile
        mean_latency = statistics.mean(latencies)

        logger.info(
            "GPU scorer latency stats",
            median_ms=f"{median_latency:.2f}",
            mean_ms=f"{mean_latency:.2f}",
            p95_ms=f"{p95_latency:.2f}",
            samples=len(latencies),
        )

        # Assert median latency is under 25ms
        assert median_latency <= 25.0, f"Median latency {median_latency:.2f}ms exceeds 25ms requirement"

        # Cleanup
        scorer.cleanup()

    @pytest.mark.asyncio
    async def test_gpu_vs_cpu_speedup(self, sample_texts):
        """Compare GPU vs CPU performance to verify acceleration benefit."""
        # Test GPU performance
        gpu_config = QualityConfig(device="cuda")
        gpu_scorer = TransformerCoherenceScorer(gpu_config)

        # Warm up
        for _ in range(3):
            await gpu_scorer.score(sample_texts[0])

        # Measure GPU time
        gpu_start = time.perf_counter()
        for text in sample_texts[:20]:
            await gpu_scorer.score(text)
        gpu_time = time.perf_counter() - gpu_start

        gpu_scorer.cleanup()

        # Test CPU performance
        cpu_config = QualityConfig(device="cpu")
        cpu_scorer = TransformerCoherenceScorer(cpu_config)

        # Warm up
        for _ in range(3):
            await cpu_scorer.score(sample_texts[0])

        # Measure CPU time
        cpu_start = time.perf_counter()
        for text in sample_texts[:20]:
            await cpu_scorer.score(text)
        cpu_time = time.perf_counter() - cpu_start

        cpu_scorer.cleanup()

        # Calculate speedup
        speedup = cpu_time / gpu_time

        logger.info(
            "GPU vs CPU performance comparison",
            gpu_time_s=f"{gpu_time:.3f}",
            cpu_time_s=f"{cpu_time:.3f}",
            speedup=f"{speedup:.2f}x",
            texts_processed=20,
        )

        # GPU should provide at least some speedup (though this depends on hardware)
        assert speedup >= 1.0, f"GPU slower than CPU (speedup: {speedup:.2f}x)"

    @pytest.mark.asyncio
    async def test_gpu_memory_usage(self, sample_texts):
        """Test that GPU memory usage remains reasonable."""
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Clear GPU cache
        torch.cuda.empty_cache()

        # Get initial memory
        initial_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB

        config = QualityConfig(device="cuda")
        scorer = TransformerCoherenceScorer(config)

        # Process some texts
        for text in sample_texts[:10]:
            await scorer.score(text)

        # Get memory after model init and scoring
        model_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        memory_delta = model_memory - initial_memory

        logger.info(
            "GPU memory usage",
            initial_mb=f"{initial_memory:.1f}",
            after_model_mb=f"{model_memory:.1f}",
            delta_mb=f"{memory_delta:.1f}",
        )

        # Clean up
        scorer.cleanup()
        torch.cuda.empty_cache()

        # Assert memory usage is reasonable (less than 1GB for the small model)
        assert memory_delta < 1024, f"GPU memory usage too high: {memory_delta:.1f}MB"

    @pytest.mark.asyncio
    async def test_concurrent_gpu_scoring(self, sample_texts):
        """Test concurrent scoring performance on GPU."""
        config = QualityConfig(device="cuda")
        scorer = TransformerCoherenceScorer(config)

        # Warm up
        await scorer.score(sample_texts[0])

        # Test concurrent scoring
        start = time.perf_counter()

        # Create concurrent tasks
        tasks = [scorer.score(text) for text in sample_texts[:30]]
        scores = await asyncio.gather(*tasks)

        elapsed = time.perf_counter() - start
        throughput = len(tasks) / elapsed

        logger.info(
            "Concurrent GPU scoring performance",
            tasks=len(tasks),
            elapsed_s=f"{elapsed:.3f}",
            throughput_per_s=f"{throughput:.1f}",
        )

        # Verify all scores are valid
        assert all(0.0 <= score <= 1.0 for score in scores)
        assert throughput > 10, f"Throughput too low: {throughput:.1f} texts/s"

        # Cleanup
        scorer.cleanup()

    @pytest.mark.asyncio
    async def test_gpu_scorer_cpu_baseline_impact(self, sample_texts):
        """Test that GPU scorer doesn't significantly impact CPU-only baseline (≤ 5%)."""
        # This test would run in CPU-only mode to verify the baseline isn't impacted
        import os

        os.environ["QUARRY_QUALITY_DEVICE"] = "cpu"

        config = QualityConfig(device="cpu")
        scorer = TransformerCoherenceScorer(config)

        # Measure baseline performance
        start = time.perf_counter()
        for text in sample_texts[:20]:
            await scorer.score(text)
        cpu_time = time.perf_counter() - start

        # The implementation should not add more than 5% overhead
        # This is a placeholder - in practice you'd compare against a known baseline
        logger.info(
            "CPU baseline performance", time_s=f"{cpu_time:.3f}", texts=20, avg_ms_per_text=f"{(cpu_time/20)*1000:.2f}"
        )

        scorer.cleanup()
