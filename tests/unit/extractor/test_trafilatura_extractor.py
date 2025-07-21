"""
Unit tests for TrafilaturaExtractor.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest
from quarrycore.extractor.trafilatura_extractor import TrafilaturaExtractor


class TestTrafilaturaExtractor:
    """Test cases for TrafilaturaExtractor."""

    def test_init(self):
        """Test extractor initialization."""
        extractor = TrafilaturaExtractor()
        assert extractor.name == "trafilatura"
        assert isinstance(extractor.config, dict)
        assert extractor.config["favor_precision"] is True

    @pytest.mark.asyncio
    async def test_extract_empty_html(self):
        """Test extraction with empty HTML."""
        extractor = TrafilaturaExtractor()
        result = await extractor.extract("")

        assert result.url is None
        assert result.text == ""
        assert result.title is None
        assert result.images == []
        assert result.language is None
        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_extract_with_url(self):
        """Test extraction with URL parameter."""
        extractor = TrafilaturaExtractor()
        test_url = "https://example.com"
        result = await extractor.extract("", url=test_url)

        assert result.url == test_url

    @pytest.mark.asyncio
    @patch("quarrycore.extractor.trafilatura_extractor.HAS_TRAFILATURA", False)
    async def test_extract_without_trafilatura(self):
        """Test extraction when trafilatura is not available."""
        extractor = TrafilaturaExtractor()
        result = await extractor.extract("<html><body>Test</body></html>")

        assert result.text == ""
        assert result.score == 0.0

    @pytest.mark.asyncio
    @patch("quarrycore.extractor.trafilatura_extractor.HAS_TRAFILATURA", True)
    @patch("quarrycore.extractor.trafilatura_extractor.trafilatura")
    async def test_extract_successful(self, mock_trafilatura):
        """Test successful extraction."""
        # Mock trafilatura functions
        mock_trafilatura.extract.return_value = "This is extracted text content."
        mock_metadata = Mock()
        mock_metadata.title = "Test Title"
        mock_metadata.language = "en"
        mock_trafilatura.extract_metadata.return_value = mock_metadata

        extractor = TrafilaturaExtractor()
        html = """
        <html>
            <head><title>Test Title</title></head>
            <body>
                <p>This is test content.</p>
                <img src="image1.jpg" alt="Image 1">
                <img src="https://example.com/image2.jpg" alt="Image 2">
            </body>
        </html>
        """

        result = await extractor.extract(html, url="https://example.com")

        assert result.text == "This is extracted text content."
        assert result.title == "Test Title"
        assert result.language == "en"
        assert result.score == 0.8
        assert "https://example.com/image1.jpg" in result.images
        assert "https://example.com/image2.jpg" in result.images

    @pytest.mark.asyncio
    @patch("quarrycore.extractor.trafilatura_extractor.HAS_TRAFILATURA", True)
    @patch("quarrycore.extractor.trafilatura_extractor.trafilatura")
    async def test_extract_no_content(self, mock_trafilatura):
        """Test extraction when trafilatura returns None."""
        mock_trafilatura.extract.return_value = None
        mock_trafilatura.extract_metadata.return_value = None

        extractor = TrafilaturaExtractor()
        result = await extractor.extract("<html><body>Test</body></html>")

        assert result.text == ""
        assert result.title is None
        assert result.score == 0.0

    @pytest.mark.asyncio
    @patch("quarrycore.extractor.trafilatura_extractor.HAS_TRAFILATURA", True)
    @patch("quarrycore.extractor.trafilatura_extractor.trafilatura")
    async def test_extract_short_content(self, mock_trafilatura):
        """Test extraction with short content."""
        mock_trafilatura.extract.return_value = "Short"
        mock_trafilatura.extract_metadata.return_value = None

        extractor = TrafilaturaExtractor()
        result = await extractor.extract("<html><body>Short</body></html>")

        assert result.text == "Short"
        assert result.score == 0.0  # Score should be 0 for short content

    @pytest.mark.asyncio
    @patch("quarrycore.extractor.trafilatura_extractor.HAS_TRAFILATURA", True)
    @patch("quarrycore.extractor.trafilatura_extractor.trafilatura")
    async def test_extract_exception_handling(self, mock_trafilatura):
        """Test exception handling during extraction."""
        mock_trafilatura.extract.side_effect = Exception("Trafilatura error")

        extractor = TrafilaturaExtractor()
        result = await extractor.extract("<html><body>Test</body></html>")

        assert result.text == ""
        assert result.score == 0.0

    @pytest.mark.asyncio
    @patch("quarrycore.extractor.trafilatura_extractor.HAS_TRAFILATURA", True)
    @patch("quarrycore.extractor.trafilatura_extractor.trafilatura")
    async def test_extract_metadata_without_title(self, mock_trafilatura):
        """Test extraction when metadata has no title."""
        mock_trafilatura.extract.return_value = "Test content"
        mock_metadata = Mock()
        mock_metadata.title = None
        mock_trafilatura.extract_metadata.return_value = mock_metadata

        extractor = TrafilaturaExtractor()
        result = await extractor.extract("<html><body>Test</body></html>")

        assert result.text == "Test content"
        assert result.title is None

    @pytest.mark.asyncio
    @patch("quarrycore.extractor.trafilatura_extractor.HAS_TRAFILATURA", True)
    @patch("quarrycore.extractor.trafilatura_extractor.trafilatura")
    async def test_extract_relative_image_urls(self, mock_trafilatura):
        """Test extraction with relative image URLs."""
        mock_trafilatura.extract.return_value = "Test content"
        mock_trafilatura.extract_metadata.return_value = None

        extractor = TrafilaturaExtractor()
        html = '<html><body><img src="relative.jpg"><img src="/absolute.jpg"></body></html>'

        result = await extractor.extract(html, url="https://example.com/page")

        assert "https://example.com/relative.jpg" in result.images
        assert "https://example.com/absolute.jpg" in result.images

    @pytest.mark.asyncio
    @patch("quarrycore.extractor.trafilatura_extractor.HAS_TRAFILATURA", True)
    @patch("quarrycore.extractor.trafilatura_extractor.trafilatura")
    async def test_extract_no_images_config(self, mock_trafilatura):
        """Test extraction when include_images is False."""
        mock_trafilatura.extract.return_value = "Test content"
        mock_trafilatura.extract_metadata.return_value = None

        extractor = TrafilaturaExtractor()
        extractor.config["include_images"] = False

        html = '<html><body><p>Test</p><img src="test.jpg"></body></html>'
        result = await extractor.extract(html)

        assert result.images == []

    def test_extract_result_validation(self):
        """Test that ExtractResult validates score range."""
        from quarrycore.extractor.models import ExtractResult

        # Valid score
        result = ExtractResult(url=None, text="Test", title=None, images=[], language=None, score=0.5)
        assert result.score == 0.5

        # Invalid score should raise ValueError
        with pytest.raises(ValueError, match="Score must be between 0.0 and 1.0"):
            ExtractResult(url=None, text="Test", title=None, images=[], language=None, score=1.5)

    def test_protocol_compliance(self):
        """Test that TrafilaturaExtractor implements the Extractor protocol."""
        from quarrycore.extractor.protocols import Extractor

        extractor = TrafilaturaExtractor()
        assert isinstance(extractor, Extractor)
        assert hasattr(extractor, "name")
        assert hasattr(extractor, "extract")
        assert extractor.name == "trafilatura"
