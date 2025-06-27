"""Tests for the extractor module."""

from __future__ import annotations

import pytest

from quarrycore.extractor import CascadeExtractor


class TestCascadeExtractor:
    """Tests for CascadeExtractor."""

    @pytest.fixture
    def extractor(self):
        """Provide test extractor instance."""
        return CascadeExtractor()

    def test_extractor_initialization(self, extractor):
        """Test extractor initialization."""
        assert extractor is not None

    def test_extract_basic_html(self, extractor, sample_html):
        """Test basic HTML content extraction."""
        # Mock implementation - will be implemented later
        pass

    def test_extract_metadata(self, extractor, sample_html):
        """Test metadata extraction from HTML."""
        # Mock implementation - will be implemented later
        pass


class TestExtractedContent:
    """Tests for ExtractedContent model."""

    def test_extraction_result_creation(self):
        """Test creating ExtractedContent instance."""
        # Mock implementation - will be implemented later
        pass
