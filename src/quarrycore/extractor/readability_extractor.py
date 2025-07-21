"""
Readability-based HTML content extractor.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

from .models import ExtractResult
from .protocols import Extractor

logger = logging.getLogger(__name__)

# Import with graceful fallback
try:
    from readability import Document  # type: ignore[import-not-found]

    HAS_READABILITY = True
except ImportError:
    Document = None
    HAS_READABILITY = False


class ReadabilityExtractor(Extractor):
    """Extractor using readability-lxml for content extraction."""

    name = "readability"

    def __init__(self) -> None:
        self.config = {
            "min_text_length": 25,
            "retry_length": 250,
            "positive_keywords": [
                "article",
                "body",
                "content",
                "entry",
                "hentry",
                "main",
                "page",
                "post",
                "text",
                "blog",
                "story",
            ],
            "negative_keywords": [
                "combx",
                "comment",
                "com-",
                "contact",
                "foot",
                "footer",
                "footnote",
                "masthead",
                "media",
                "meta",
                "outbrain",
                "promo",
                "related",
                "scroll",
                "shoutbox",
                "sidebar",
                "sponsor",
                "shopping",
                "tags",
                "tool",
                "widget",
            ],
        }

    async def extract(self, html: str, *, url: str | None = None) -> ExtractResult:
        """Extract content using readability-lxml.

        Args:
            html: HTML content to extract from
            url: Optional URL for context

        Returns:
            ExtractResult with extracted content
        """
        if not HAS_READABILITY or not html.strip():
            logger.warning("Readability not available or empty HTML")
            return ExtractResult(
                url=url,
                text="",
                title=None,
                images=[],
                language=None,
                score=0.0,
            )

        try:
            # Run readability in thread pool since it's CPU-intensive
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self._extract_sync,
                html,
                url,
            )
            return result
        except Exception as e:
            logger.warning(f"Readability extraction failed: {e}")
            return ExtractResult(
                url=url,
                text="",
                title=None,
                images=[],
                language=None,
                score=0.0,
            )

    def _extract_sync(self, html: str, url: str | None) -> ExtractResult:
        """Synchronous extraction using readability-lxml."""
        if not Document:
            return ExtractResult(
                url=url,
                text="",
                title=None,
                images=[],
                language=None,
                score=0.0,
            )

        try:
            # Create readability document
            doc = Document(
                html,
                url=url,
                min_text_length=self.config["min_text_length"],
                retry_length=self.config["retry_length"],
                positive_keywords=self.config["positive_keywords"],
                negative_keywords=self.config["negative_keywords"],
            )

            # Extract title
            title = doc.short_title() or doc.title()

            # Extract main content
            content_html = doc.summary()

            # Convert HTML to text
            text = self._html_to_text(content_html)

            # Extract images from content
            images = []
            if content_html:
                img_pattern = r'<img[^>]+src=["\']([^"\']+)["\'][^>]*>'
                images = re.findall(img_pattern, content_html, re.IGNORECASE)
                # Convert relative URLs to absolute if possible
                if url:
                    from urllib.parse import urljoin

                    images = [
                        urljoin(url, img) if not img.startswith(("http://", "https://")) else img for img in images
                    ]

            return ExtractResult(
                url=url,
                text=text,
                title=title if title else None,
                images=images,
                language=None,  # readability doesn't provide language detection
                score=0.7 if text and len(text.strip()) > 20 else 0.0,
            )

        except Exception as e:
            logger.warning(f"Readability sync extraction failed: {e}")
            return ExtractResult(
                url=url,
                text="",
                title=None,
                images=[],
                language=None,
                score=0.0,
            )

    def _html_to_text(self, html: str) -> str:
        """Convert HTML to plain text."""
        if not html:
            return ""

        try:
            # Try to use lxml for better text extraction
            try:
                from lxml import etree  # type: ignore[import-not-found]
                from lxml import html as lxml_html

                doc = lxml_html.fromstring(html)
                text = etree.tostring(doc, method="text", encoding="unicode")
                return " ".join(text.split())
            except ImportError:
                # Fallback to regex-based cleaning
                pass

            # Remove script and style elements
            html = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
            html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE)

            # Remove HTML tags
            text = re.sub(r"<[^>]+>", " ", html)

            # Clean up whitespace
            text = " ".join(text.split())

            return text

        except Exception as e:
            logger.warning(f"HTML to text conversion failed: {e}")
            return ""
