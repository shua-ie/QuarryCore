from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from quarrycore.protocols import ContentMetadata, ExtractedContent, QualityScore

logger = logging.getLogger(__name__)


class GrammarScorer:
    """
    Scores content based on grammar and spelling using LanguageTool.
    Falls back to heuristic scoring if LanguageTool is unavailable.
    """

    def __init__(self, language: str = "en-US"):
        """Initializes the GrammarScorer."""
        self._tool: Optional[object] = None
        self._language = language
        self._fallback_mode = False

        try:
            import language_tool_python  # type: ignore

            self._tool = language_tool_python.LanguageTool(language)
            logger.info("LanguageTool initialized successfully")
        except (ImportError, RuntimeError, Exception) as e:
            # RuntimeError occurs when Java is not available or wrong version
            logger.warning(f"LanguageTool initialization failed: {e}. Using fallback scoring.")
            self._fallback_mode = True

    async def score(
        self,
        content: ExtractedContent,
        metadata: ContentMetadata,
        score: QualityScore,
    ) -> None:
        """
        Checks grammar and updates the QualityScore object.

        Args:
            content: The extracted content of the document.
            metadata: The metadata of the document.
            score: The QualityScore object to be updated in place.
        """
        if not content.text or content.word_count < 10:
            score.grammar_score = 0.0
            score.quality_factors["grammar_score"] = 0.0
            score.quality_factors["grammar_errors"] = 0
            return

        if self._fallback_mode:
            # Use heuristic scoring when LanguageTool is unavailable
            grammar_score = self._heuristic_grammar_score(content.text)
            score.grammar_score = grammar_score
            score.quality_factors["grammar_score"] = grammar_score
            score.quality_factors["grammar_errors"] = -1.0  # Indicate fallback mode
            return

        text = content.text

        try:
            # language-tool-python's check is CPU-bound
            if self._tool and hasattr(self._tool, "check"):
                matches = await asyncio.to_thread(self._tool.check, text)  # type: ignore
            else:
                matches = []

            error_count = len(matches)

            # Simple scoring: 1 is perfect, 0 is bad.
            # Penalize based on errors per 100 words.
            errors_per_100_words = (error_count / content.word_count) * 100

            # Cap penalty at 1.0
            grammar_score = max(0.0, 1.0 - (errors_per_100_words / 10.0))

            score.grammar_score = grammar_score
            score.quality_factors["grammar_score"] = grammar_score
            score.quality_factors["grammar_errors"] = float(error_count)
        except Exception as e:
            logger.warning(f"Grammar check failed: {e}. Using fallback.")
            # Fall back to heuristic scoring
            grammar_score = self._heuristic_grammar_score(content.text)
            score.grammar_score = grammar_score
            score.quality_factors["grammar_score"] = grammar_score
            score.quality_factors["grammar_errors"] = -1.0

    def _heuristic_grammar_score(self, text: str) -> float:
        """
        Simple heuristic-based grammar scoring when LanguageTool is unavailable.

        Checks for:
        - Basic punctuation patterns
        - Sentence structure
        - Common grammar indicators
        """
        score = 1.0

        # Check for basic sentence structure
        sentences = text.split(".")
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # Penalize sentences without capital letters at start
            if sentence and not sentence[0].isupper():
                score -= 0.05

            # Check for very short or very long sentences
            words = sentence.split()
            if len(words) < 3 or len(words) > 50:
                score -= 0.05

        # Check for common grammar issues (simplified)
        # Double spaces
        if "  " in text:
            score -= 0.1

        # Missing space after punctuation
        import re

        if re.search(r"[,.!?][a-zA-Z]", text):
            score -= 0.1

        # Multiple punctuation
        if re.search(r"[.!?]{2,}", text):
            score -= 0.05

        # Check grammar using language_tool_python if available
        if self._tool and hasattr(self._tool, "check"):
            try:
                matches = self._tool.check(text)  # type: ignore
                grammar_errors = len(matches)

                # Basic grammar score based on error density
                error_density = grammar_errors / max(1, len(text.split()))
                max(0.0, 1.0 - (error_density * 10))

            except Exception as e:
                logger.warning(f"Grammar check error: {e}")

        # Ensure score stays in valid range
        return max(0.0, min(1.0, score))
