from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import textstat

if TYPE_CHECKING:
    from quarrycore.protocols import ContentMetadata, ExtractedContent, QualityScore


def calculate_lexical_diversity(text: str) -> float:
    """Calculates lexical diversity (type-token ratio)."""
    words = text.split()
    if not words:
        return 0.0
    return len(set(words)) / len(words)


class LexicalScorer:
    """
    Scores content based on lexical metrics like readability, complexity,
    and diversity.
    """

    async def score(
        self,
        content: ExtractedContent,
        metadata: ContentMetadata,
        score: QualityScore,
    ) -> None:
        """
        Calculates lexical metrics and updates the QualityScore object.

        Args:
            content: The extracted content of the document.
            metadata: The metadata of the document.
            score: The QualityScore object to be updated in place.
        """
        if not content.text or content.word_count < 10:  # Skip for very short texts
            score.quality_factors["lexical_score"] = 0.0
            return

        text = content.text

        # Run synchronous textstat functions in a thread pool
        readability_score, avg_sentence_length, diversity_score = await asyncio.gather(
            asyncio.to_thread(textstat.flesch_reading_ease, text),
            asyncio.to_thread(textstat.avg_sentence_length, text),
            asyncio.to_thread(calculate_lexical_diversity, text),
        )

        # Update the main QualityScore object
        score.readability_score = float(readability_score)

        # Update the detailed breakdown in quality_factors
        score.quality_factors["flesch_reading_ease"] = float(readability_score)
        score.quality_factors["avg_sentence_length"] = float(avg_sentence_length)
        score.quality_factors["lexical_diversity"] = float(diversity_score)

        # A simple aggregated lexical score (can be refined)
        # Normalize scores to a 0-1 range if they aren't already
        # Flesch is 0-100, higher is better.
        normalized_readability = min(max(readability_score, 0), 100) / 100.0
        # Diversity is 0-1, higher is better.
        # Sentence length - let's say 15-25 is ideal. Penalize extremes.
        normalized_sent_len = 1.0 - min(abs(avg_sentence_length - 20.0) / 20.0, 1.0)

        lexical_score = (normalized_readability * 0.5) + (diversity_score * 0.3) + (normalized_sent_len * 0.2)
        score.quality_factors["lexical_score"] = lexical_score
