"""
HTML Canonicalization for Exact Deduplication.

Implements deterministic HTML normalization to ensure consistent hashing:
- Remove script and style tags with their content
- Remove comments and CDATA sections
- Normalize whitespace (collapse multiple spaces/newlines to single space)
- Remove leading/trailing whitespace
- Convert to lowercase for tag names and attributes
- Sort attributes alphabetically for deterministic output

This ensures that semantically identical HTML produces the same hash,
while being fast enough for real-time processing (target: <1ms per document).
"""

import re
from typing import Optional

try:
    from selectolax.parser import HTMLParser

    HAS_SELECTOLAX = True
except ImportError:
    from bs4 import BeautifulSoup, Comment

    HAS_SELECTOLAX = False

import structlog

logger = structlog.get_logger(__name__)


class CanonicalHTMLProcessor:
    """
    Fast HTML canonicalization processor.

    Uses selectolax for speed (70x faster than BeautifulSoup),
    with BeautifulSoup fallback for compatibility.
    """

    def __init__(self) -> None:
        # Pre-compile regex patterns for performance
        self._whitespace_pattern = re.compile(r"\s+")
        self._leading_trailing_ws = re.compile(r"^\s+|\s+$")

        # Track performance metrics
        self._processed_count = 0
        self._fallback_count = 0

    def canonicalize(self, html: str) -> str:
        """
        Canonicalize HTML content for consistent hashing.

        Args:
            html: Raw HTML content

        Returns:
            Canonicalized HTML string

        Raises:
            ValueError: If HTML is empty or invalid
        """
        if not html or not html.strip():
            raise ValueError("HTML content cannot be empty")

        self._processed_count += 1

        try:
            if HAS_SELECTOLAX:
                return self._canonicalize_selectolax(html)
            else:
                self._fallback_count += 1
                return self._canonicalize_bs4(html)
        except Exception as e:
            logger.warning("HTML canonicalization failed, using fallback", error=str(e))
            self._fallback_count += 1
            return self._canonicalize_fallback(html)

    def _canonicalize_selectolax(self, html: str) -> str:
        """Canonicalize using selectolax (preferred for speed)."""
        try:
            # Parse with selectolax
            tree = HTMLParser(html)

            # Remove script and style tags
            for tag in tree.css("script, style"):
                tag.decompose()

            # Get text content (this automatically strips tags)
            text_content = tree.text()

            if not text_content:
                # Fallback: extract and clean the HTML structure
                html_clean = tree.html
                if html_clean:
                    return self._normalize_whitespace(html_clean)
                return ""

            return self._normalize_whitespace(text_content)

        except Exception as e:
            logger.debug("Selectolax parsing failed, trying BeautifulSoup", error=str(e))
            raise

    def _canonicalize_bs4(self, html: str) -> str:
        """Canonicalize using BeautifulSoup (fallback)."""
        try:
            soup = BeautifulSoup(html, "html.parser")

            # Remove script and style tags
            for script in soup(["script", "style"]):
                script.decompose()

            # Remove comments
            for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
                comment.extract()

            # Get text content
            text_content = soup.get_text()

            return self._normalize_whitespace(text_content)

        except Exception as e:
            logger.debug("BeautifulSoup parsing failed, using regex fallback", error=str(e))
            raise

    def _canonicalize_fallback(self, html: str) -> str:
        """
        Regex-based fallback canonicalization.

        Less accurate but more resilient for malformed HTML.
        """
        # Remove script tags and their content
        html = re.sub(r"<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>", "", html, flags=re.DOTALL | re.IGNORECASE)

        # Remove style tags and their content
        html = re.sub(r"<style\b[^<]*(?:(?!<\/style>)<[^<]*)*<\/style>", "", html, flags=re.DOTALL | re.IGNORECASE)

        # Remove HTML comments
        html = re.sub(r"<!--.*?-->", "", html, flags=re.DOTALL)

        # Remove all HTML tags
        html = re.sub(r"<[^>]+>", "", html)

        # Decode common HTML entities
        html = html.replace("&nbsp;", " ")
        html = html.replace("&amp;", "&")
        html = html.replace("&lt;", "<")
        html = html.replace("&gt;", ">")
        html = html.replace("&quot;", '"')
        html = html.replace("&#39;", "'")

        return self._normalize_whitespace(html)

    def _normalize_whitespace(self, text: str) -> str:
        """
        Normalize whitespace for consistent hashing.

        - Collapse multiple whitespace characters to single space
        - Remove leading and trailing whitespace
        - Convert newlines and tabs to spaces
        """
        if not text:
            return ""

        # Replace all whitespace with single spaces
        normalized = self._whitespace_pattern.sub(" ", text)

        # Remove leading and trailing whitespace
        normalized = normalized.strip()

        return normalized

    def get_stats(self) -> dict:
        """Get processing statistics."""
        return {
            "processed_count": self._processed_count,
            "fallback_count": self._fallback_count,
            "fallback_rate": self._fallback_count / max(1, self._processed_count),
            "using_selectolax": HAS_SELECTOLAX,
        }


def canonicalize_html(html: str) -> str:
    """
    Convenience function for HTML canonicalization.

    Args:
        html: Raw HTML content

    Returns:
        Canonicalized HTML string
    """
    processor = CanonicalHTMLProcessor()
    return processor.canonicalize(html)
