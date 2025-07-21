"""
Unit tests for ExtractorManager with comprehensive branch coverage.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from quarrycore.config.config import ExtractionSettings
from quarrycore.extractor.manager import ExtractorManager
from quarrycore.extractor.models import ExtractResult
from quarrycore.protocols import QualityScore


@pytest.fixture
def mock_quality_assessor():
    """Create a mock quality assessor."""
    assessor = AsyncMock()
    # Default to returning good quality
    assessor.assess_quality.return_value = QualityScore(overall_score=0.8, quality_factors={"test": 0.8})
    return assessor


@pytest.fixture
def default_settings():
    """Create default extraction settings."""
    return ExtractionSettings(
        cascade_order=["trafilatura", "readability", "soup_fallback"],
        quality_threshold=0.6,
        domain_overrides={
            "nytimes.com": ["readability", "soup_fallback"],
            "wikipedia.org": ["soup_fallback", "trafilatura"],
        },
    )


@pytest.fixture
def mock_extractors():
    """Create mock extractors."""
    extractors = {}

    # Trafilatura mock
    trafilatura = AsyncMock()
    trafilatura.name = "trafilatura"
    trafilatura.extract.return_value = ExtractResult(
        url="http://example.com",
        text="High quality content from trafilatura",
        title="Test Title",
        images=[],
        language="en",
        score=0.0,  # Will be set by quality assessor
    )
    extractors["trafilatura"] = trafilatura

    # Readability mock
    readability = AsyncMock()
    readability.name = "readability"
    readability.extract.return_value = ExtractResult(
        url="http://example.com",
        text="Medium quality content from readability",
        title="Test Title",
        images=[],
        language="en",
        score=0.0,
    )
    extractors["readability"] = readability

    # Soup fallback mock
    soup = AsyncMock()
    soup.name = "soup_fallback"
    soup.extract.return_value = ExtractResult(
        url="http://example.com",
        text="Low quality content from soup",
        title="Test Title",
        images=[],
        language="en",
        score=0.0,
    )
    extractors["soup_fallback"] = soup

    return extractors


class TestExtractorManager:
    """Test cases for ExtractorManager."""

    @pytest.mark.asyncio
    async def test_happy_path_first_extractor_good(self, mock_quality_assessor, default_settings, mock_extractors):
        """Test successful extraction with first extractor producing good quality."""
        # Patch the extractors
        with patch.multiple(
            "quarrycore.extractor.manager",
            TrafilaturaExtractor=lambda: mock_extractors["trafilatura"],
            ReadabilityExtractor=lambda: mock_extractors["readability"],
            SoupFallbackExtractor=lambda: mock_extractors["soup_fallback"],
        ):
            manager = ExtractorManager(mock_quality_assessor, default_settings)

            # Quality assessor returns good score
            mock_quality_assessor.assess_quality.return_value.overall_score = 0.8

            result = await manager.extract("http://example.com", "<html>test</html>")

            assert result is not None
            assert result.text == "High quality content from trafilatura"
            assert result.score == 0.8

            # Only first extractor should be called
            mock_extractors["trafilatura"].extract.assert_called_once()
            mock_extractors["readability"].extract.assert_not_called()
            mock_extractors["soup_fallback"].extract.assert_not_called()

    @pytest.mark.asyncio
    async def test_fallback_after_low_quality(self, mock_quality_assessor, default_settings, mock_extractors):
        """Test fallback to next extractor when quality is below threshold."""
        with patch.multiple(
            "quarrycore.extractor.manager",
            TrafilaturaExtractor=lambda: mock_extractors["trafilatura"],
            ReadabilityExtractor=lambda: mock_extractors["readability"],
            SoupFallbackExtractor=lambda: mock_extractors["soup_fallback"],
        ):
            manager = ExtractorManager(mock_quality_assessor, default_settings)

            # Set up quality scores: trafilatura low, readability good
            quality_scores = [0.4, 0.7]  # Below and above threshold
            mock_quality_assessor.assess_quality.side_effect = [
                QualityScore(overall_score=score, quality_factors={}) for score in quality_scores
            ]

            result = await manager.extract("http://example.com", "<html>test</html>")

            assert result is not None
            assert result.text == "Medium quality content from readability"
            assert result.score == 0.7

            # Both extractors should be called
            mock_extractors["trafilatura"].extract.assert_called_once()
            mock_extractors["readability"].extract.assert_called_once()
            mock_extractors["soup_fallback"].extract.assert_not_called()

    @pytest.mark.asyncio
    async def test_extractor_raises_exception_fallback(self, mock_quality_assessor, default_settings, mock_extractors):
        """Test fallback when extractor raises exception."""
        with patch.multiple(
            "quarrycore.extractor.manager",
            TrafilaturaExtractor=lambda: mock_extractors["trafilatura"],
            ReadabilityExtractor=lambda: mock_extractors["readability"],
            SoupFallbackExtractor=lambda: mock_extractors["soup_fallback"],
        ):
            manager = ExtractorManager(mock_quality_assessor, default_settings)

            # First extractor raises exception
            mock_extractors["trafilatura"].extract.side_effect = RuntimeError("Extraction failed")

            # Second extractor returns good content
            mock_quality_assessor.assess_quality.return_value.overall_score = 0.8

            result = await manager.extract("http://example.com", "<html>test</html>")

            assert result is not None
            assert result.text == "Medium quality content from readability"
            assert result.score == 0.8

            # Both extractors should be called
            mock_extractors["trafilatura"].extract.assert_called_once()
            mock_extractors["readability"].extract.assert_called_once()

    @pytest.mark.asyncio
    async def test_threshold_edge_exact(self, mock_quality_assessor, default_settings, mock_extractors):
        """Test extraction with quality score exactly at threshold."""
        with patch.multiple(
            "quarrycore.extractor.manager",
            TrafilaturaExtractor=lambda: mock_extractors["trafilatura"],
            ReadabilityExtractor=lambda: mock_extractors["readability"],
            SoupFallbackExtractor=lambda: mock_extractors["soup_fallback"],
        ):
            manager = ExtractorManager(mock_quality_assessor, default_settings)

            # Quality exactly at threshold (0.6)
            mock_quality_assessor.assess_quality.return_value.overall_score = 0.6

            result = await manager.extract("http://example.com", "<html>test</html>")

            assert result is not None
            assert result.score == 0.6  # Should pass at exact threshold

            # Only first extractor needed
            mock_extractors["trafilatura"].extract.assert_called_once()

    @pytest.mark.asyncio
    async def test_domain_override_order(self, mock_quality_assessor, default_settings, mock_extractors):
        """Test domain-specific extractor ordering."""
        with patch.multiple(
            "quarrycore.extractor.manager",
            TrafilaturaExtractor=lambda: mock_extractors["trafilatura"],
            ReadabilityExtractor=lambda: mock_extractors["readability"],
            SoupFallbackExtractor=lambda: mock_extractors["soup_fallback"],
        ):
            manager = ExtractorManager(mock_quality_assessor, default_settings)

            # Good quality from first extractor in override order
            mock_quality_assessor.assess_quality.return_value.overall_score = 0.8

            # Test nytimes.com override
            result = await manager.extract("https://www.nytimes.com/article", "<html>test</html>")

            assert result is not None
            assert result.text == "Medium quality content from readability"

            # Readability should be called first due to override
            mock_extractors["readability"].extract.assert_called_once()
            mock_extractors["trafilatura"].extract.assert_not_called()

    @pytest.mark.asyncio
    async def test_all_extractors_fail(self, mock_quality_assessor, default_settings, mock_extractors):
        """Test when all extractors fail or return low quality."""
        with patch.multiple(
            "quarrycore.extractor.manager",
            TrafilaturaExtractor=lambda: mock_extractors["trafilatura"],
            ReadabilityExtractor=lambda: mock_extractors["readability"],
            SoupFallbackExtractor=lambda: mock_extractors["soup_fallback"],
        ):
            manager = ExtractorManager(mock_quality_assessor, default_settings)

            # All return low quality
            mock_quality_assessor.assess_quality.return_value.overall_score = 0.3

            result = await manager.extract("http://example.com", "<html>test</html>")

            assert result is None

            # All extractors should be tried
            mock_extractors["trafilatura"].extract.assert_called_once()
            mock_extractors["readability"].extract.assert_called_once()
            mock_extractors["soup_fallback"].extract.assert_called_once()

    @pytest.mark.asyncio
    async def test_empty_content_skip(self, mock_quality_assessor, default_settings, mock_extractors):
        """Test that extractors returning empty content are skipped."""
        with patch.multiple(
            "quarrycore.extractor.manager",
            TrafilaturaExtractor=lambda: mock_extractors["trafilatura"],
            ReadabilityExtractor=lambda: mock_extractors["readability"],
            SoupFallbackExtractor=lambda: mock_extractors["soup_fallback"],
        ):
            manager = ExtractorManager(mock_quality_assessor, default_settings)

            # First extractor returns empty content
            mock_extractors["trafilatura"].extract.return_value = ExtractResult(
                url="http://example.com",
                text="",  # Empty
                title="Test",
                images=[],
                language="en",
                score=0.0,
            )

            # Second returns good content
            mock_quality_assessor.assess_quality.return_value.overall_score = 0.8

            result = await manager.extract("http://example.com", "<html>test</html>")

            assert result is not None
            assert result.text == "Medium quality content from readability"

            # Both should be called
            mock_extractors["trafilatura"].extract.assert_called_once()
            mock_extractors["readability"].extract.assert_called_once()

    @pytest.mark.asyncio
    async def test_subdomain_matching(self, mock_quality_assessor, default_settings, mock_extractors):
        """Test subdomain matching for domain overrides."""
        with patch.multiple(
            "quarrycore.extractor.manager",
            TrafilaturaExtractor=lambda: mock_extractors["trafilatura"],
            ReadabilityExtractor=lambda: mock_extractors["readability"],
            SoupFallbackExtractor=lambda: mock_extractors["soup_fallback"],
        ):
            manager = ExtractorManager(mock_quality_assessor, default_settings)

            mock_quality_assessor.assess_quality.return_value.overall_score = 0.8

            # Test subdomain matching
            result = await manager.extract("https://en.wikipedia.org/wiki/Test", "<html>test</html>")

            assert result is not None
            assert result.text == "Low quality content from soup"

            # Soup should be called first due to wikipedia.org override
            mock_extractors["soup_fallback"].extract.assert_called_once()
            mock_extractors["trafilatura"].extract.assert_not_called()

    @pytest.mark.asyncio
    async def test_invalid_url_fallback_to_default(self, mock_quality_assessor, default_settings, mock_extractors):
        """Test fallback to default order when URL parsing fails."""
        with patch.multiple(
            "quarrycore.extractor.manager",
            TrafilaturaExtractor=lambda: mock_extractors["trafilatura"],
            ReadabilityExtractor=lambda: mock_extractors["readability"],
            SoupFallbackExtractor=lambda: mock_extractors["soup_fallback"],
        ):
            manager = ExtractorManager(mock_quality_assessor, default_settings)

            mock_quality_assessor.assess_quality.return_value.overall_score = 0.8

            # Invalid URL should use default order
            result = await manager.extract("not-a-valid-url", "<html>test</html>")

            assert result is not None
            assert result.text == "High quality content from trafilatura"

            # Default order: trafilatura first
            mock_extractors["trafilatura"].extract.assert_called_once()

    def test_invalid_cascade_order_validation(self, mock_quality_assessor):
        """Test validation of invalid extractor names in cascade order."""
        settings = ExtractionSettings(cascade_order=["invalid_extractor", "trafilatura"], quality_threshold=0.6)

        with pytest.raises(ValueError, match="Invalid extractor 'invalid_extractor'"):
            ExtractorManager(mock_quality_assessor, settings)

    def test_invalid_domain_override_validation(self, mock_quality_assessor):
        """Test validation of invalid extractor names in domain overrides."""
        settings = ExtractionSettings(
            cascade_order=["trafilatura"],
            quality_threshold=0.6,
            domain_overrides={"example.com": ["invalid_extractor"]},
        )

        with pytest.raises(ValueError, match="Invalid extractor 'invalid_extractor'"):
            ExtractorManager(mock_quality_assessor, settings)

    @pytest.mark.asyncio
    async def test_get_metrics(self, mock_quality_assessor, default_settings, mock_extractors):
        """Test metrics collection."""
        with patch.multiple(
            "quarrycore.extractor.manager",
            TrafilaturaExtractor=lambda: mock_extractors["trafilatura"],
            ReadabilityExtractor=lambda: mock_extractors["readability"],
            SoupFallbackExtractor=lambda: mock_extractors["soup_fallback"],
        ):
            manager = ExtractorManager(mock_quality_assessor, default_settings)

            # Run some extractions
            mock_quality_assessor.assess_quality.return_value.overall_score = 0.8
            await manager.extract("http://example.com", "<html>test</html>")

            # Low quality to trigger fallback
            mock_quality_assessor.assess_quality.side_effect = [
                QualityScore(overall_score=0.3, quality_factors={}),
                QualityScore(overall_score=0.8, quality_factors={}),
            ]
            await manager.extract("http://example.com", "<html>test2</html>")

            metrics = manager.get_metrics()

            assert metrics["trafilatura"]["attempts"] == 2
            assert metrics["trafilatura"]["successes"] == 1
            assert metrics["trafilatura"]["success_rate"] == 0.5
            assert metrics["readability"]["attempts"] == 1
            assert metrics["readability"]["successes"] == 1
            assert metrics["readability"]["success_rate"] == 1.0
            assert metrics["soup_fallback"]["attempts"] == 0
