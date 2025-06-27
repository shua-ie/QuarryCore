from __future__ import annotations

from typing import Protocol

from quarrycore.protocols import ContentMetadata, ExtractedContent, QualityScore


class Scorer(Protocol):
    """Protocol for a component that contributes to the overall quality score."""

    async def score(
        self,
        content: ExtractedContent,
        metadata: ContentMetadata,
        score: QualityScore,
    ) -> None:
        """
        Calculates a specific aspect of quality and updates the QualityScore object.

        Args:
            content: The extracted content of the document.
            metadata: The metadata of the document.
            score: The QualityScore object to be updated in place.
        """
        ...
