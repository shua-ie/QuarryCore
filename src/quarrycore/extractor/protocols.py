"""
Protocols for pluggable HTML extraction strategies.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .models import ExtractResult


@runtime_checkable
class Extractor(Protocol):
    """Pluggable HTML-to-ExtractResult strategy."""

    name: str

    async def extract(self, html: str, *, url: str | None = None) -> ExtractResult:
        """Extract content from HTML string.

        Args:
            html: HTML content to extract from
            url: Optional URL for context

        Returns:
            ExtractResult with extracted content
        """
        ...
