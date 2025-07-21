"""
Quality assessment orchestrator that aggregates multiple scorers.
"""

from __future__ import annotations

import asyncio
from typing import Optional

import structlog

from quarrycore.config.config import QualityConfig
from quarrycore.observability.metrics import METRICS
from quarrycore.quality.scorers import (
    ALL_SCORERS,
    LanguageScorer,
    LengthScorer,
    TransformerCoherenceScorer,
)

logger = structlog.get_logger(__name__)


class QualityAssessor:
    """
    Orchestrates quality assessment using multiple scorers.
    Implements singleton pattern to avoid multiple GPU model loads.
    """

    _instance: Optional[QualityAssessor] = None
    _lock = asyncio.Lock()

    def __new__(cls, config: QualityConfig):
        """Singleton pattern to ensure only one instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config: QualityConfig):
        """Initialize the assessor with scorers."""
        # Only initialize once
        if hasattr(self, "_initialized"):
            return

        self.config = config
        self.scorers = []
        self._weights = {}

        # Initialize scorers
        self.length_scorer = LengthScorer()
        self.scorers.append(self.length_scorer)
        self._weights[self.length_scorer.name] = self.length_scorer.WEIGHT

        self.language_scorer = LanguageScorer()
        self.scorers.append(self.language_scorer)
        self._weights[self.language_scorer.name] = self.language_scorer.WEIGHT

        self.coherence_scorer = TransformerCoherenceScorer(config)
        self.scorers.append(self.coherence_scorer)
        self._weights[self.coherence_scorer.name] = self.coherence_scorer.WEIGHT

        # Verify weights sum to 1.0
        total_weight = sum(self._weights.values())
        if abs(total_weight - 1.0) > 0.001:
            logger.warning("Scorer weights don't sum to 1.0", total_weight=total_weight, weights=self._weights)

        self._initialized = True
        logger.info("QualityAssessor initialized", weights=self._weights)

    async def score(self, text: str) -> float:
        """
        Score text quality using all scorers.

        Args:
            text: The text to score

        Returns:
            Aggregated quality score between 0 and 1
        """
        if not text:
            return 0.0

        # Run all scorers concurrently
        tasks = []
        for scorer in self.scorers:
            tasks.append(self._score_with_metrics(scorer, text))

        # Gather results
        scores = await asyncio.gather(*tasks, return_exceptions=True)

        # Aggregate scores
        total_score = 0.0
        valid_weights = 0.0

        for i, score_result in enumerate(scores):
            scorer = self.scorers[i]

            if isinstance(score_result, Exception):
                logger.error("Scorer failed", scorer=scorer.name, error=str(score_result))
                METRICS["quality_scorer_errors"].labels(scorer=scorer.name).inc()
                continue

            # Add weighted score
            weight = self._weights[scorer.name]
            total_score += score_result.value * weight
            valid_weights += weight

            logger.debug("Scorer result", scorer=scorer.name, score=score_result.value, weight=weight)

        # Normalize if not all scorers succeeded
        if valid_weights > 0 and valid_weights < 1.0:
            total_score = total_score / valid_weights

        # Clamp to [0, 1]
        final_score = max(0.0, min(1.0, total_score))

        logger.info("Quality assessment complete", score=final_score, text_length=len(text))

        return final_score

    async def _score_with_metrics(self, scorer, text: str):
        """Score with metrics tracking."""
        scorer_name = scorer.name

        with METRICS["quality_scorer_latency"].labels(scorer=scorer_name).time():
            result = await scorer(text)

        return result

    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, "coherence_scorer"):
            self.coherence_scorer.cleanup()
