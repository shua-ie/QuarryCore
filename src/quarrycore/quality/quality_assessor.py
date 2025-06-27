from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, AsyncGenerator, List

from quarrycore.config.config import QualityConfig
from quarrycore.protocols import ContentMetadata, DomainType, ExtractedContent, QualityProtocol, QualityScore

from .grammar_scorer import GrammarScorer
from .heuristic_scorer import HeuristicScorer
from .lexical_scorer import LexicalScorer
from .neural_scorer import NeuralScorer

if TYPE_CHECKING:
    from .scorer import Scorer


class QualityAssessor(QualityProtocol):
    """
    Orchestrates the entire quality assessment pipeline, running various scorers
    and aggregating their results into a final quality score.
    """

    def __init__(self, config: QualityConfig):
        """
        Initializes the QualityAssessor with scorers and configuration.

        Args:
            config: The quality configuration object.
        """
        self.config = config
        self.lexical_scorer = LexicalScorer()
        self.grammar_scorer = GrammarScorer()
        self.heuristic_scorer = HeuristicScorer(
            # In a real system, this would be loaded from a more elaborate config
            domain_keywords={}
        )

        self.neural_scorer = NeuralScorer()

        self.scorers: List[Scorer] = [
            self.lexical_scorer,
            self.grammar_scorer,
            self.heuristic_scorer,
        ]

    async def assess_quality(
        self,
        content: ExtractedContent,
        metadata: ContentMetadata,
        **kwargs: Any,
    ) -> QualityScore:
        """Assesses the quality of a single document."""
        async for result in self.assess_batch([(content, metadata)]):
            return result
        # Fallback - should never reach here
        return QualityScore()

    async def assess_batch(
        self,
        content_batch: list[tuple[ExtractedContent, ContentMetadata]],
        *,
        batch_size: int = 32,
        use_gpu: bool = True,
    ) -> AsyncGenerator[QualityScore, None]:
        """Assesses the quality of a batch of documents."""

        # Process in chunks based on batch_size
        for i in range(0, len(content_batch), batch_size):
            chunk = content_batch[i : i + batch_size]

            # Initialize QualityScore objects for this chunk
            scores = [QualityScore() for _ in chunk]

            # --- Run Scorers Concurrently ---

            # Handle neural scorer separately for batching efficiency
            neural_task = self.neural_scorer.score_batch([content for content, metadata in chunk])

            # Run other scorers
            other_scorers_tasks = []
            for scorer in self.scorers:
                for j, (content, metadata) in enumerate(chunk):
                    task = scorer.score(content, metadata, scores[j])
                    other_scorers_tasks.append(task)

            # Gather all results
            gathered_results = await asyncio.gather(neural_task, *other_scorers_tasks)

            neural_results = gathered_results[0]

            # Apply neural scores
            for j, result in enumerate(neural_results):
                scores[j].coherence_score = result.get("coherence_score", 0.0)
                scores[j].toxicity_score = result.get("toxicity_score", 0.0)
                scores[j].quality_factors.update(result.get("quality_factors", {}))

            # --- Calculate Final Scores and Yield ---
            for j, (_content, metadata) in enumerate(chunk):
                self._calculate_overall_score(scores[j], metadata)
                yield scores[j]

    async def detect_bias(self, text: str) -> float:
        """Detects potential bias in content."""
        # This is a placeholder implementation.
        # A real implementation would use a dedicated bias detection model.
        return 0.0

    async def detect_toxicity(self, text: str) -> float:
        """Detects toxic content."""
        # This is a placeholder implementation.
        # A real implementation would use a dedicated toxicity detection model.
        return 0.0

    async def assess_domain_relevance(self, content: str, domain: DomainType) -> float:
        """Assess relevance to a specific domain."""
        # This is a placeholder implementation.
        # A real implementation would use a dedicated domain relevance model.
        return 0.5

    def _calculate_overall_score(self, score: QualityScore, metadata: ContentMetadata) -> None:
        """Calculates the final aggregated score based on domain-specific weights."""
        domain_config = self.config.domains.get(metadata.domain_type)
        if domain_config is None:
            domain_config = self.config.default

        weights = domain_config.weights

        total_score = 0.0

        # Aggregate scores from quality_factors using weights
        # This is a flexible way to add new scores
        total_score += score.quality_factors.get("lexical_score", 0.0) * weights.get("lexical", 0.0)
        total_score += score.quality_factors.get("grammar_score", 0.0) * weights.get("grammar", 0.0)
        total_score += score.quality_factors.get("neural_coherence", 0.0) * weights.get("neural_coherence", 0.0)
        total_score += score.quality_factors.get("information_density", 0.0) * weights.get(
            "heuristic_info_density", 0.0
        )
        total_score += score.quality_factors.get("domain_relevance", 0.0) * weights.get(
            "heuristic_domain_relevance", 0.0
        )

        # Negative weights for negative signals
        total_score += score.quality_factors.get("toxicity", 0.0) * weights.get("toxicity", 0.0)
        total_score += score.quality_factors.get("spam_score", 0.0) * weights.get("heuristic_spam", 0.0)

        score.overall_score = max(0.0, min(1.0, total_score))

    async def assess_text(self, text: str) -> float:
        """
        Protocol compliance method - assesses quality of raw text.

        This method exists to match validation expectations.
        Returns a quality score between 0.0 and 1.0.
        """
        from quarrycore.protocols import ContentMetadata, DomainType, ExtractedContent

        # Create minimal ExtractedContent and ContentMetadata for assessment
        content = ExtractedContent(text=text)
        metadata = ContentMetadata(url=f"text_assessment_{hash(text)}", domain_type=DomainType.GENERAL)

        quality_score = await self.assess_quality(content, metadata)
        return quality_score.overall_score
