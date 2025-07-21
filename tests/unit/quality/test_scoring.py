"""Unit tests for quality scoring system."""

from __future__ import annotations

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import structlog
from quarrycore.config.config import QualityConfig
from quarrycore.quality.assessor import QualityAssessor
from quarrycore.quality.scorers import LanguageScorer, LengthScorer, Score, TransformerCoherenceScorer

logger = structlog.get_logger(__name__)


class TestLengthScorer:
    """Test LengthScorer functionality."""

    @pytest.mark.asyncio
    async def test_length_scorer_short_text(self):
        """Test that short text gets score 0.0."""
        scorer = LengthScorer()
        score = await scorer("This is a short text.")

        assert isinstance(score, Score)
        assert score.name == "length"
        assert score.value == 0.0

    @pytest.mark.asyncio
    async def test_length_scorer_long_text(self):
        """Test that text > 400 chars gets score 1.0."""
        scorer = LengthScorer()
        long_text = "a" * 401  # 401 characters
        score = await scorer(long_text)

        assert isinstance(score, Score)
        assert score.name == "length"
        assert score.value == 1.0

    @pytest.mark.asyncio
    async def test_length_scorer_exact_boundary(self):
        """Test boundary condition at exactly 400 chars."""
        scorer = LengthScorer()
        boundary_text = "a" * 400  # Exactly 400 characters
        score = await scorer(boundary_text)

        assert score.value == 0.0  # Should be 0.0 since condition is > 400


class TestLanguageScorer:
    """Test LanguageScorer functionality."""

    @pytest.mark.asyncio
    async def test_language_scorer_english_text(self):
        """Test that English text gets score 1.0."""
        scorer = LanguageScorer()
        english_text = "This is a clear example of English text with proper grammar and vocabulary."
        score = await scorer(english_text)

        assert isinstance(score, Score)
        assert score.name == "language"
        assert score.value == 1.0

    @pytest.mark.asyncio
    async def test_language_scorer_non_english_text(self):
        """Test that non-English text gets score 0.0."""
        scorer = LanguageScorer()
        spanish_text = "Este es un ejemplo claro de texto en español con gramática y vocabulario adecuados."
        score = await scorer(spanish_text)

        assert score.name == "language"
        assert score.value == 0.0

    @pytest.mark.asyncio
    async def test_language_scorer_empty_text(self):
        """Test that empty text gets score 0.0."""
        scorer = LanguageScorer()
        score = await scorer("")

        assert score.value == 0.0

    @pytest.mark.asyncio
    async def test_language_scorer_short_text(self):
        """Test that very short text gets handled properly."""
        scorer = LanguageScorer()
        score = await scorer("Hi")

        # Short text might not be detected as English by mock
        assert score.value in [0.0, 1.0]


class TestTransformerCoherenceScorer:
    """Test TransformerCoherenceScorer functionality."""

    @pytest.mark.asyncio
    async def test_coherence_scorer_initialization(self):
        """Test scorer initialization."""
        config = QualityConfig(device="cpu")
        scorer = TransformerCoherenceScorer(config)

        assert scorer.config == config
        assert scorer._model is None
        assert not scorer._initialized

    @pytest.mark.asyncio
    async def test_coherence_scorer_test_mode(self):
        """Test scorer in test mode."""
        os.environ["QUARRY_TEST_MODE"] = "1"
        try:
            config = QualityConfig(device="cpu")
            scorer = TransformerCoherenceScorer(config)
            score = await scorer.score("Any text in test mode")

            assert score == 0.75  # Mock score in test mode
        finally:
            os.environ.pop("QUARRY_TEST_MODE", None)

    @pytest.mark.asyncio
    async def test_coherence_scorer_short_text(self):
        """Test that very short text gets score 0.0."""
        os.environ["QUARRY_TEST_MODE"] = "1"
        try:
            config = QualityConfig(device="cpu")
            scorer = TransformerCoherenceScorer(config)
            score = await scorer.score("Too short")

            assert score == 0.0  # Less than 10 words
        finally:
            os.environ.pop("QUARRY_TEST_MODE", None)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("device", ["cpu", "cuda", "auto"])
    async def test_coherence_scorer_device_settings(self, device):
        """Test different device settings."""
        config = QualityConfig(device=device)
        scorer = TransformerCoherenceScorer(config)

        assert scorer.config.device == device


class TestQualityAssessor:
    """Test QualityAssessor functionality."""

    @pytest.mark.asyncio
    async def test_assessor_singleton(self):
        """Test that assessor implements singleton pattern."""
        config = QualityConfig()
        assessor1 = QualityAssessor(config)
        assessor2 = QualityAssessor(config)

        assert assessor1 is assessor2

    @pytest.mark.asyncio
    async def test_assessor_initialization(self):
        """Test assessor initialization."""
        config = QualityConfig()
        assessor = QualityAssessor(config)

        assert len(assessor.scorers) == 3
        assert assessor._weights["length"] == 0.3
        assert assessor._weights["language"] == 0.4
        assert assessor._weights["coherence"] == 0.3
        assert sum(assessor._weights.values()) == 1.0

    @pytest.mark.asyncio
    async def test_assessor_empty_text(self):
        """Test assessor with empty text."""
        config = QualityConfig()
        assessor = QualityAssessor(config)
        score = await assessor.score("")

        assert score == 0.0

    @pytest.mark.asyncio
    async def test_assessor_good_english_text(self):
        """Test assessor with good English text."""
        os.environ["QUARRY_TEST_MODE"] = "1"
        try:
            config = QualityConfig()
            # Clear singleton
            QualityAssessor._instance = None
            assessor = QualityAssessor(config)

            good_text = (
                """
            This is a well-written article about artificial intelligence and its impact
            on modern society. The article discusses various applications of AI in healthcare,
            education, and transportation. It provides clear examples and maintains a logical
            flow throughout. The content is informative and engaging, suitable for a general
            audience interested in technology trends. The writing style is professional yet
            accessible, making complex concepts easy to understand. Overall, this represents
            high-quality content that would be valuable for training language models.
            """
                * 2
            )  # Make it longer than 400 chars

            score = await assessor.score(good_text)

            # Length: 1.0 (> 400 chars) * 0.3 = 0.3
            # Language: 1.0 (English) * 0.4 = 0.4
            # Coherence: 0.75 (test mode) * 0.3 = 0.225
            # Total: 0.3 + 0.4 + 0.225 = 0.925
            assert score == pytest.approx(0.925, rel=0.01)
        finally:
            os.environ.pop("QUARRY_TEST_MODE", None)
            QualityAssessor._instance = None

    @pytest.mark.asyncio
    async def test_assessor_bad_text(self):
        """Test assessor with poor quality text."""
        os.environ["QUARRY_TEST_MODE"] = "1"
        try:
            config = QualityConfig()
            # Clear singleton
            QualityAssessor._instance = None
            assessor = QualityAssessor(config)

            bad_text = "Short non-English texto mezclado with bad quality"

            score = await assessor.score(bad_text)

            # Length: 0.0 (< 400 chars) * 0.3 = 0.0
            # Language: 0.0 (mixed/non-English) * 0.4 = 0.0
            # Coherence: 0.0 (< 10 words) * 0.3 = 0.0
            # Total: 0.0
            assert score == 0.0
        finally:
            os.environ.pop("QUARRY_TEST_MODE", None)
            QualityAssessor._instance = None

    @pytest.mark.asyncio
    async def test_assessor_scorer_failure_handling(self):
        """Test that assessor handles scorer failures gracefully."""
        config = QualityConfig()
        # Clear singleton
        QualityAssessor._instance = None
        assessor = QualityAssessor(config)

        # Mock a scorer to fail
        failing_scorer = AsyncMock()
        failing_scorer.name = "failing"
        failing_scorer.side_effect = Exception("Scorer failed")

        # Replace one scorer with failing one
        assessor.scorers[0] = failing_scorer
        assessor._weights["failing"] = 0.3

        score = await assessor.score("Any text")

        # Should still return a score from other scorers
        assert 0.0 <= score <= 1.0


@pytest.mark.asyncio
async def test_integration_all_scorers():
    """Integration test with all scorers working together."""
    os.environ["QUARRY_TEST_MODE"] = "1"
    try:
        config = QualityConfig(device="cpu", min_score=0.5)

        # Clear singleton
        QualityAssessor._instance = None
        assessor = QualityAssessor(config)

        test_cases = [
            # (text, expected_min_score, expected_max_score)
            ("", 0.0, 0.0),  # Empty text
            ("Short text", 0.0, 0.3),  # Short, might be English
            ("This is a longer English text " * 20, 0.6, 1.0),  # Long English text
            ("Dies ist ein längerer deutscher Text " * 20, 0.2, 0.6),  # Long German text
        ]

        for text, min_score, max_score in test_cases:
            score = await assessor.score(text)
            assert (
                min_score <= score <= max_score
            ), f"Score {score} not in range [{min_score}, {max_score}] for text: {text[:50]}..."

    finally:
        os.environ.pop("QUARRY_TEST_MODE", None)
        QualityAssessor._instance = None
