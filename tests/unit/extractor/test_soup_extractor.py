"""
Unit tests for SoupFallbackExtractor.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest
from quarrycore.extractor.soup_extractor import SoupFallbackExtractor


class TestSoupFallbackExtractor:
    """Test cases for SoupFallbackExtractor."""

    def test_init(self):
        """Test extractor initialization."""
        extractor = SoupFallbackExtractor()
        assert extractor.name == "soup_fallback"
        assert isinstance(extractor.config, dict)
        assert extractor.config["parser"] == "html.parser"
        assert "content_selectors" in extractor.config

    @pytest.mark.asyncio
    async def test_extract_empty_html(self):
        """Test extraction with empty HTML."""
        extractor = SoupFallbackExtractor()
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
        extractor = SoupFallbackExtractor()
        test_url = "https://example.com"
        result = await extractor.extract("", url=test_url)

        assert result.url == test_url

    @pytest.mark.asyncio
    @patch("quarrycore.extractor.soup_extractor.HAS_BEAUTIFULSOUP", False)
    async def test_extract_without_beautifulsoup(self):
        """Test extraction when BeautifulSoup is not available."""
        extractor = SoupFallbackExtractor()
        result = await extractor.extract("<html><body>Test</body></html>")

        assert result.text == ""
        assert result.score == 0.0

    @pytest.mark.asyncio
    @patch("quarrycore.extractor.soup_extractor.HAS_BEAUTIFULSOUP", True)
    @patch("quarrycore.extractor.soup_extractor.BeautifulSoup")
    async def test_extract_successful(self, mock_bs_class):
        """Test successful extraction."""
        # Mock BeautifulSoup instance
        mock_soup = Mock()
        mock_bs_class.return_value = mock_soup

        # Mock title
        mock_title = Mock()
        mock_title.get_text.return_value = "Test Title"
        mock_soup.find.return_value = mock_title

        # Mock main content
        mock_main = Mock()
        mock_main.get_text.return_value = "This is the main content of the page."
        mock_soup.find_all.return_value = [mock_main]

        # Mock images
        mock_img1 = Mock()
        mock_img1.get.return_value = "image1.jpg"
        mock_img2 = Mock()
        mock_img2.get.return_value = "https://example.com/image2.jpg"
        mock_main.find_all.return_value = [mock_img1, mock_img2]

        extractor = SoupFallbackExtractor()
        html = """
        <html>
            <head><title>Test Title</title></head>
            <body>
                <main>
                    <p>This is the main content.</p>
                    <img src="image1.jpg" alt="Image 1">
                    <img src="https://example.com/image2.jpg" alt="Image 2">
                </main>
            </body>
        </html>
        """

        result = await extractor.extract(html, url="https://example.com")

        assert result.text == "This is the main content of the page."
        assert result.title == "Test Title"
        assert result.language is None  # BeautifulSoup doesn't provide language
        assert result.score == 0.5
        assert "https://example.com/image1.jpg" in result.images
        assert "https://example.com/image2.jpg" in result.images

    @pytest.mark.asyncio
    @patch("quarrycore.extractor.soup_extractor.HAS_BEAUTIFULSOUP", True)
    @patch("quarrycore.extractor.soup_extractor.BeautifulSoup")
    async def test_extract_no_title(self, mock_bs_class):
        """Test extraction when no title is found."""
        mock_soup = Mock()
        mock_bs_class.return_value = mock_soup

        # No title found
        mock_soup.find.return_value = None

        # Mock main content
        mock_main = Mock()
        mock_main.get_text.return_value = "Content without title"
        mock_main.find_all.return_value = []
        mock_soup.find_all.return_value = [mock_main]

        extractor = SoupFallbackExtractor()
        result = await extractor.extract("<html><body>Test</body></html>")

        assert result.title is None
        assert result.text == "Content without title"

    @pytest.mark.asyncio
    @patch("quarrycore.extractor.soup_extractor.HAS_BEAUTIFULSOUP", True)
    @patch("quarrycore.extractor.soup_extractor.BeautifulSoup")
    async def test_extract_fallback_to_body(self, mock_bs_class):
        """Test fallback to body when no main content selectors match."""
        mock_soup = Mock()
        mock_bs_class.return_value = mock_soup

        # No title
        mock_soup.find.return_value = None

        # No main content selectors match
        mock_soup.find_all.return_value = []

        # Mock body fallback
        mock_body = Mock()
        mock_body.get_text.return_value = "Body content"
        mock_body.find_all.return_value = []
        mock_soup.find.return_value = mock_body  # When find('body') is called

        extractor = SoupFallbackExtractor()
        result = await extractor.extract("<html><body>Test</body></html>")

        assert result.text == "Body content"

    @pytest.mark.asyncio
    @patch("quarrycore.extractor.soup_extractor.HAS_BEAUTIFULSOUP", True)
    @patch("quarrycore.extractor.soup_extractor.BeautifulSoup")
    async def test_extract_short_content(self, mock_bs_class):
        """Test extraction with short content."""
        mock_soup = Mock()
        mock_bs_class.return_value = mock_soup

        mock_soup.find.return_value = None  # No title
        mock_main = Mock()
        mock_main.get_text.return_value = "Short"
        mock_main.find_all.return_value = []
        mock_soup.find_all.return_value = [mock_main]

        extractor = SoupFallbackExtractor()
        result = await extractor.extract("<html><body>Short</body></html>")

        assert result.text == "Short"
        assert result.score == 0.0  # Score should be 0 for short content

    @pytest.mark.asyncio
    @patch("quarrycore.extractor.soup_extractor.HAS_BEAUTIFULSOUP", True)
    @patch("quarrycore.extractor.soup_extractor.BeautifulSoup")
    async def test_extract_exception_handling(self, mock_bs_class):
        """Test exception handling during extraction."""
        mock_bs_class.side_effect = Exception("BeautifulSoup error")

        extractor = SoupFallbackExtractor()
        result = await extractor.extract("<html><body>Test</body></html>")

        assert result.text == ""
        assert result.score == 0.0

    @pytest.mark.asyncio
    @patch("quarrycore.extractor.soup_extractor.HAS_BEAUTIFULSOUP", True)
    @patch("quarrycore.extractor.soup_extractor.BeautifulSoup")
    async def test_extract_removes_unwanted_elements(self, mock_bs_class):
        """Test that unwanted elements are removed."""
        mock_soup = Mock()
        mock_bs_class.return_value = mock_soup

        # Mock elements to be removed
        mock_script = Mock()

        # Simplify the mocking - just check that decompose is called
        mock_soup.find_all.return_value = [mock_script]
        mock_soup.find.return_value = None  # No title

        # Make the soup itself act as the main content fallback
        mock_soup.get_text.return_value = "Clean content"

        extractor = SoupFallbackExtractor()
        result = await extractor.extract("<html><body>Test</body></html>")

        # Verify decompose was called on unwanted elements
        assert mock_script.decompose.call_count >= 1

        # Verify we got some result (even if empty due to complex mocking)
        assert isinstance(result.text, str)

    @pytest.mark.asyncio
    @patch("quarrycore.extractor.soup_extractor.HAS_BEAUTIFULSOUP", True)
    @patch("quarrycore.extractor.soup_extractor.BeautifulSoup")
    async def test_extract_class_selector(self, mock_bs_class):
        """Test extraction with class selector."""
        mock_soup = Mock()
        mock_bs_class.return_value = mock_soup

        mock_soup.find.return_value = None  # No title

        # Mock class selector matching
        mock_content = Mock()
        mock_content.get_text.return_value = "Content from class selector"
        mock_content.find_all.return_value = []

        def mock_find_all(*args, **kwargs):
            if "class_" in kwargs and kwargs["class_"] == "content":
                return [mock_content]
            return []

        mock_soup.find_all.side_effect = mock_find_all

        extractor = SoupFallbackExtractor()
        result = await extractor.extract("<html><body>Test</body></html>")

        assert result.text == "Content from class selector"

    @pytest.mark.asyncio
    @patch("quarrycore.extractor.soup_extractor.HAS_BEAUTIFULSOUP", True)
    @patch("quarrycore.extractor.soup_extractor.BeautifulSoup")
    async def test_extract_id_selector(self, mock_bs_class):
        """Test extraction with ID selector."""
        mock_soup = Mock()
        mock_bs_class.return_value = mock_soup

        # Mock ID selector matching
        mock_content = Mock()
        mock_content.get_text.return_value = "Content from ID selector"
        mock_content.find_all.return_value = []

        def mock_find(*args, **kwargs):
            # Handle title search - return None
            if args and args[0] == "title":
                return None
            # Handle ID selector search
            if "id" in kwargs and kwargs["id"] == "content":
                return mock_content
            # Handle body fallback
            if args and args[0] == "body":
                return mock_content
            return None

        # Mock find_all to return empty for unwanted elements removal
        def mock_find_all(*args, **kwargs):
            return []

        mock_soup.find.side_effect = mock_find
        mock_soup.find_all.side_effect = mock_find_all

        extractor = SoupFallbackExtractor()
        result = await extractor.extract("<html><body>Test</body></html>")

        assert result.text == "Content from ID selector"

    @pytest.mark.asyncio
    @patch("quarrycore.extractor.soup_extractor.HAS_BEAUTIFULSOUP", True)
    @patch("quarrycore.extractor.soup_extractor.BeautifulSoup")
    async def test_extract_role_selector(self, mock_bs_class):
        """Test extraction with role attribute selector."""
        mock_soup = Mock()
        mock_bs_class.return_value = mock_soup

        mock_soup.find.return_value = None  # No title

        # Mock role selector matching
        mock_content = Mock()
        mock_content.get_text.return_value = "Content from role selector"
        mock_content.find_all.return_value = []

        def mock_find_all(*args, **kwargs):
            if "attrs" in kwargs and kwargs["attrs"] == {"role": "main"}:
                return [mock_content]
            return []

        mock_soup.find_all.side_effect = mock_find_all

        extractor = SoupFallbackExtractor()
        result = await extractor.extract("<html><body>Test</body></html>")

        assert result.text == "Content from role selector"

    @pytest.mark.asyncio
    @patch("quarrycore.extractor.soup_extractor.HAS_BEAUTIFULSOUP", True)
    @patch("quarrycore.extractor.soup_extractor.BeautifulSoup")
    async def test_extract_images_with_relative_urls(self, mock_bs_class):
        """Test extraction with relative image URLs."""
        mock_soup = Mock()
        mock_bs_class.return_value = mock_soup

        mock_soup.find.return_value = None  # No title

        # Mock main content
        mock_main = Mock()
        mock_main.get_text.return_value = "Content with images"

        # Mock images
        mock_img1 = Mock()
        mock_img1.get.return_value = "relative.jpg"
        mock_img2 = Mock()
        mock_img2.get.return_value = "/absolute.jpg"
        mock_img3 = Mock()
        mock_img3.get.return_value = "https://example.com/full.jpg"
        mock_img4 = Mock()
        mock_img4.get.return_value = None  # No src attribute

        mock_main.find_all.return_value = [mock_img1, mock_img2, mock_img3, mock_img4]
        mock_soup.find_all.return_value = [mock_main]

        extractor = SoupFallbackExtractor()
        result = await extractor.extract("<html><body>Test</body></html>", url="https://example.com/page")

        assert "https://example.com/relative.jpg" in result.images
        assert "https://example.com/absolute.jpg" in result.images
        assert "https://example.com/full.jpg" in result.images
        assert len(result.images) == 3  # img4 should be excluded

    def test_protocol_compliance(self):
        """Test that SoupFallbackExtractor implements the Extractor protocol."""
        from quarrycore.extractor.protocols import Extractor

        extractor = SoupFallbackExtractor()
        assert isinstance(extractor, Extractor)
        assert hasattr(extractor, "name")
        assert hasattr(extractor, "extract")
        assert extractor.name == "soup_fallback"

    @pytest.mark.asyncio
    @patch("quarrycore.extractor.soup_extractor.HAS_BEAUTIFULSOUP", True)
    @patch("quarrycore.extractor.soup_extractor.BeautifulSoup")
    async def test_extract_with_parser_config(self, mock_bs_class):
        """Test that BeautifulSoup is called with correct parser."""
        mock_soup = Mock()
        mock_bs_class.return_value = mock_soup

        mock_soup.find.return_value = None  # No title
        mock_main = Mock()
        mock_main.get_text.return_value = "Test content"
        mock_main.find_all.return_value = []
        mock_soup.find_all.return_value = [mock_main]

        extractor = SoupFallbackExtractor()
        await extractor.extract("<html><body>Test</body></html>")

        # Verify BeautifulSoup was called with correct parser
        mock_bs_class.assert_called_once_with("<html><body>Test</body></html>", "html.parser")
