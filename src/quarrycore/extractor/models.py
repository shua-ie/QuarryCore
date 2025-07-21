"""
Data models for extraction results.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True, frozen=True)
class ExtractResult:
    """Result of HTML content extraction."""

    url: str | None
    text: str
    title: str | None
    images: list[str]
    language: str | None
    score: float  # 0-1 – populated later by QualityAssessor

    def __post_init__(self) -> None:
        """Validate the result."""
        if not (0.0 <= self.score <= 1.0):
            raise ValueError("Score must be between 0.0 and 1.0")
