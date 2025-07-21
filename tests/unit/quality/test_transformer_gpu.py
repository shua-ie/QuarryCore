"""
Unit tests for TransformerCoherenceScorer GPU functionality.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from quarrycore.config.config import QualityConfig
from quarrycore.quality.scorers import TransformerCoherenceScorer


@pytest.mark.no_cuda_required
class TestTransformerGPUPath:
    """Test GPU code paths without requiring actual CUDA hardware."""

    @pytest.fixture
    def quality_config_cuda(self):
        """Create a QualityConfig with CUDA device setting."""
        config = QualityConfig(device="cuda")
        return config

    @pytest.fixture
    def quality_config_auto(self):
        """Create a QualityConfig with auto device setting."""
        config = QualityConfig(device="auto")
        return config

    @pytest.fixture
    def quality_config_cpu(self):
        """Create a QualityConfig with CPU device setting."""
        config = QualityConfig(device="cpu")
        return config

    @pytest.mark.asyncio
    async def test_gpu_path_when_cuda_available(self, quality_config_cuda, monkeypatch):
        """Test that GPU path is taken when CUDA is available."""
        # Mock torch to simulate CUDA availability
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.device.return_value.type = "cuda"

        # Mock SentenceTransformer
        mock_model = MagicMock()
        mock_model.encode.return_value = [[0.5] * 384]  # Mock embedding
        mock_model.max_seq_length = 256

        with patch("quarrycore.quality.scorers.torch", mock_torch), patch(
            "quarrycore.quality.scorers.SentenceTransformer", return_value=mock_model
        ), patch("quarrycore.quality.scorers.HAS_ML_LIBS", True), patch(
            "quarrycore.quality.scorers.cosine_similarity", return_value=[[0.8]]
        ):
            scorer = TransformerCoherenceScorer(quality_config_cuda)
            score = await scorer.score("This is a test text for coherence scoring.")

            # Verify CUDA device was requested
            mock_torch.device.assert_called_with("cuda")
            assert score == 0.8

    @pytest.mark.asyncio
    async def test_cpu_fallback_when_cuda_not_available(self, quality_config_cuda, monkeypatch):
        """Test that CPU fallback works when CUDA is requested but not available."""
        # Mock torch with CUDA not available
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.device.return_value.type = "cpu"

        mock_model = MagicMock()
        mock_model.encode.return_value = [[0.5] * 384]
        mock_model.max_seq_length = 256

        with patch("quarrycore.quality.scorers.torch", mock_torch), patch(
            "quarrycore.quality.scorers.SentenceTransformer", return_value=mock_model
        ), patch("quarrycore.quality.scorers.HAS_ML_LIBS", True), patch(
            "quarrycore.quality.scorers.cosine_similarity", return_value=[[0.7]]
        ):
            scorer = TransformerCoherenceScorer(quality_config_cuda)
            score = await scorer.score("Test text")

            # Verify CPU device was used as fallback
            mock_torch.device.assert_called_with("cpu")
            assert score == 0.7

    @pytest.mark.asyncio
    async def test_auto_device_selection_with_cuda(self, quality_config_auto):
        """Test auto device selection when CUDA is available."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.device.return_value.type = "cuda"

        mock_model = MagicMock()
        mock_model.encode.return_value = [[0.5] * 384]
        mock_model.max_seq_length = 256

        with patch("quarrycore.quality.scorers.torch", mock_torch), patch(
            "quarrycore.quality.scorers.SentenceTransformer", return_value=mock_model
        ), patch("quarrycore.quality.scorers.HAS_ML_LIBS", True), patch(
            "quarrycore.quality.scorers.cosine_similarity", return_value=[[0.85]]
        ):
            scorer = TransformerCoherenceScorer(quality_config_auto)
            score = await scorer.score("Test text")

            # Verify CUDA was auto-selected
            mock_torch.device.assert_called_with("cuda")
            assert score == 0.85

    @pytest.mark.asyncio
    async def test_explicit_cpu_device(self, quality_config_cpu):
        """Test explicit CPU device selection."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True  # CUDA available but CPU requested
        mock_torch.device.return_value.type = "cpu"

        mock_model = MagicMock()
        mock_model.encode.return_value = [[0.5] * 384]
        mock_model.max_seq_length = 256

        with patch("quarrycore.quality.scorers.torch", mock_torch), patch(
            "quarrycore.quality.scorers.SentenceTransformer", return_value=mock_model
        ), patch("quarrycore.quality.scorers.HAS_ML_LIBS", True), patch(
            "quarrycore.quality.scorers.cosine_similarity", return_value=[[0.75]]
        ):
            scorer = TransformerCoherenceScorer(quality_config_cpu)
            score = await scorer.score("Test text")

            # Verify CPU was explicitly used despite CUDA availability
            mock_torch.device.assert_called_with("cpu")
            assert score == 0.75

    @pytest.mark.asyncio
    async def test_model_to_device_called(self, quality_config_cuda):
        """Test that model is moved to the correct device."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_device = MagicMock()
        mock_device.type = "cuda"
        mock_torch.device.return_value = mock_device

        mock_model = MagicMock()
        mock_model.encode.return_value = [[0.5] * 384]
        mock_model.max_seq_length = 256

        mock_sentence_transformer = MagicMock(return_value=mock_model)

        with patch("quarrycore.quality.scorers.torch", mock_torch), patch(
            "quarrycore.quality.scorers.SentenceTransformer", mock_sentence_transformer
        ), patch("quarrycore.quality.scorers.HAS_ML_LIBS", True), patch(
            "quarrycore.quality.scorers.cosine_similarity", return_value=[[0.8]]
        ):
            scorer = TransformerCoherenceScorer(quality_config_cuda)
            score = await scorer.score("Test text")

            # Verify SentenceTransformer was called with device parameter
            mock_sentence_transformer.assert_called_with("sentence-transformers/all-MiniLM-L6-v2", device=mock_device)

            # Verify score is valid
            assert score == 0.8

    @pytest.mark.asyncio
    async def test_cleanup_clears_cuda_cache(self, quality_config_cuda):
        """Test that cleanup clears CUDA cache when using GPU."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.empty_cache = MagicMock()
        mock_device = MagicMock()
        mock_device.type = "cuda"
        mock_torch.device.return_value = mock_device

        mock_model = MagicMock()
        mock_model.encode.return_value = [[0.5] * 384]

        with patch("quarrycore.quality.scorers.torch", mock_torch), patch(
            "quarrycore.quality.scorers.SentenceTransformer", return_value=mock_model
        ), patch("quarrycore.quality.scorers.HAS_ML_LIBS", True), patch(
            "quarrycore.quality.scorers.cosine_similarity", return_value=[[0.8]]
        ):
            scorer = TransformerCoherenceScorer(quality_config_cuda)
            score = await scorer.score("Test text")  # Initialize the model

            # Verify score is valid
            assert 0.0 <= score <= 1.0

            scorer.cleanup()

            # Verify CUDA cache was cleared
            mock_torch.cuda.empty_cache.assert_called_once()

    @pytest.mark.asyncio
    async def test_test_mode_returns_mock_score(self, quality_config_cuda, monkeypatch):
        """Test that test mode returns mock score without loading models."""
        monkeypatch.setenv("QUARRY_TEST_MODE", "1")

        # Import after setting env var to ensure _TEST_MODE is set
        from quarrycore.quality.scorers import TransformerCoherenceScorer

        scorer = TransformerCoherenceScorer(quality_config_cuda)
        score = await scorer.score("Test text")

        # Test mode should return 0.75
        assert score == 0.75

    @pytest.mark.asyncio
    async def test_short_text_returns_zero(self, quality_config_cpu):
        """Test that short text returns zero score."""
        mock_torch = MagicMock()
        mock_model = MagicMock()

        with patch("quarrycore.quality.scorers.torch", mock_torch), patch(
            "quarrycore.quality.scorers.SentenceTransformer", return_value=mock_model
        ), patch("quarrycore.quality.scorers.HAS_ML_LIBS", True):
            scorer = TransformerCoherenceScorer(quality_config_cpu)

            # Test with short text
            score = await scorer.score("Hi")
            assert score == 0.0

            # Test with empty text
            score = await scorer.score("")
            assert score == 0.0
