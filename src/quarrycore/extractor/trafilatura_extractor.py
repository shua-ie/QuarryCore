"""
Trafilatura-based HTML content extractor.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from .models import ExtractResult
from .protocols import Extractor

logger = logging.getLogger(__name__)

# Import with graceful fallback
try:
    import trafilatura  # type: ignore[import-not-found]

    HAS_TRAFILATURA = True
except ImportError:
    trafilatura = None
    HAS_TRAFILATURA = False


class TrafilaturaExtractor(Extractor):
    """Extractor using Trafilatura library for high-precision content extraction."""

    name = "trafilatura"

    def __init__(self) -> None:
        self.config = {
            "favor_precision": True,
            "include_comments": False,
            "include_tables": True,
            "include_images": True,
            "include_formatting": True,
            "include_links": True,
        }

    async def extract(self, html: str, *, url: str | None = None) -> ExtractResult:
        """Extract content using Trafilatura.

        Args:
            html: HTML content to extract from
            url: Optional URL for context

        Returns:
            ExtractResult with extracted content
        """
        if not HAS_TRAFILATURA or not html.strip():
            logger.warning("Trafilatura not available or empty HTML")
            return ExtractResult(
                url=url,
                text="",
                title=None,
                images=[],
                language=None,
                score=0.0,
            )

        try:
            # Run trafilatura in thread pool since it's CPU-intensive
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self._extract_sync,
                html,
                url,
            )
            return result
        except Exception as e:
            logger.warning(f"Trafilatura extraction failed: {e}")
            return ExtractResult(
                url=url,
                text="",
                title=None,
                images=[],
                language=None,
                score=0.0,
            )

    def _extract_sync(self, html: str, url: str | None) -> ExtractResult:
        """Synchronous extraction using Trafilatura."""
        if not trafilatura:
            return ExtractResult(
                url=url,
                text="",
                title=None,
                images=[],
                language=None,
                score=0.0,
            )

        try:
            # Extract main content
            extracted_text = trafilatura.extract(
                html,
                url=url,
                favor_precision=self.config["favor_precision"],
                include_comments=self.config["include_comments"],
                include_tables=self.config["include_tables"],
                include_images=self.config["include_images"],
                include_formatting=self.config["include_formatting"],
                include_links=self.config["include_links"],
            )

            # Extract metadata
            metadata = trafilatura.extract_metadata(html, default_url=url)

            # Extract images
            images = []
            if self.config["include_images"]:
                # Basic image extraction - trafilatura doesn't provide direct image URLs
                import re

                img_pattern = r'<img[^>]+src=["\']([^"\']+)["\'][^>]*>'
                images = re.findall(img_pattern, html, re.IGNORECASE)
                # Convert relative URLs to absolute if possible
                if url:
                    from urllib.parse import urljoin

                    images = [
                        urljoin(url, img) if not img.startswith(("http://", "https://")) else img for img in images
                    ]

            return ExtractResult(
                url=url,
                text=extracted_text or "",
                title=metadata.title if metadata and hasattr(metadata, "title") else None,
                images=images,
                language=metadata.language if metadata and hasattr(metadata, "language") else None,
                score=0.8 if extracted_text and len(extracted_text.strip()) > 20 else 0.0,
            )

        except Exception as e:
            logger.warning(f"Trafilatura sync extraction failed: {e}")
            return ExtractResult(
                url=url,
                text="",
                title=None,
                images=[],
                language=None,
                score=0.0,
            )
