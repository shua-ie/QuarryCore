"""
BeautifulSoup-based HTML content extractor as fallback.
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
    from bs4 import BeautifulSoup  # type: ignore[import-not-found]

    HAS_BEAUTIFULSOUP = True
except ImportError:
    BeautifulSoup = None
    HAS_BEAUTIFULSOUP = False


class SoupFallbackExtractor(Extractor):
    """Extractor using BeautifulSoup as a last-chance fallback."""

    name = "soup_fallback"

    def __init__(self) -> None:
        self.config = {
            "parser": "html.parser",  # Use built-in parser for maximum compatibility
            "content_selectors": [
                "main",
                "article",
                ".content",
                "#content",
                ".post",
                ".entry",
                ".article-body",
                '[role="main"]',
                ".main-content",
                ".post-content",
                ".entry-content",
            ],
            "remove_tags": ["script", "style", "nav", "header", "footer", "aside", "noscript"],
            "remove_classes": ["nav", "navigation", "menu", "sidebar", "ad", "advertisement", "footer", "header"],
        }

    async def extract(self, html: str, *, url: str | None = None) -> ExtractResult:
        """Extract content using BeautifulSoup.

        Args:
            html: HTML content to extract from
            url: Optional URL for context

        Returns:
            ExtractResult with extracted content
        """
        if not HAS_BEAUTIFULSOUP or not html.strip():
            logger.warning("BeautifulSoup not available or empty HTML")
            return ExtractResult(
                url=url,
                text="",
                title=None,
                images=[],
                language=None,
                score=0.0,
            )

        try:
            # Run BeautifulSoup in thread pool since it can be CPU-intensive
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self._extract_sync,
                html,
                url,
            )
            return result
        except Exception as e:
            logger.warning(f"BeautifulSoup extraction failed: {e}")
            return ExtractResult(
                url=url,
                text="",
                title=None,
                images=[],
                language=None,
                score=0.0,
            )

    def _extract_sync(self, html: str, url: str | None) -> ExtractResult:
        """Synchronous extraction using BeautifulSoup."""
        if not BeautifulSoup:
            return ExtractResult(
                url=url,
                text="",
                title=None,
                images=[],
                language=None,
                score=0.0,
            )

        try:
            # Parse HTML
            soup = BeautifulSoup(html, self.config["parser"])

            # Extract title
            title = None
            title_tag = soup.find("title")
            if title_tag:
                title = title_tag.get_text(strip=True)

            # Remove unwanted elements
            for tag_name in self.config["remove_tags"]:
                for tag in soup.find_all(tag_name):
                    tag.decompose()

            # Remove elements with unwanted classes
            for class_name in self.config["remove_classes"]:
                for element in soup.find_all(class_=class_name):
                    element.decompose()

            # Try to find main content using selectors
            main_content = None
            for selector in self.config["content_selectors"]:
                if selector.startswith("."):
                    # Class selector
                    elements = soup.find_all(class_=selector[1:])
                elif selector.startswith("#"):
                    # ID selector
                    elements = [soup.find(id=selector[1:])]
                elif selector.startswith("["):
                    # Attribute selector (simplified)
                    if selector == '[role="main"]':
                        elements = soup.find_all(attrs={"role": "main"})
                    else:
                        elements = []
                else:
                    # Tag selector
                    elements = soup.find_all(selector)

                if elements and elements[0]:
                    main_content = elements[0]
                    break

            # Fallback to body if no main content found
            if not main_content:
                main_content = soup.find("body") or soup

            # Extract text content
            text = main_content.get_text(separator=" ", strip=True)

            # Extract images
            images = []
            img_tags = main_content.find_all("img")  # type: ignore[attr-defined]
            for img_tag in img_tags:
                src = img_tag.get("src")  # type: ignore[attr-defined]
                if src and isinstance(src, str):
                    # Convert relative URLs to absolute if possible
                    if url and not src.startswith(("http://", "https://")):
                        from urllib.parse import urljoin

                        src = urljoin(url, src)
                    images.append(src)

            return ExtractResult(
                url=url,
                text=text,
                title=title,
                images=images,
                language=None,  # BeautifulSoup doesn't provide language detection
                score=0.5 if text and len(text.strip()) > 20 else 0.0,
            )

        except Exception as e:
            logger.warning(f"BeautifulSoup sync extraction failed: {e}")
            return ExtractResult(
                url=url,
                text="",
                title=None,
                images=[],
                language=None,
                score=0.0,
            )
