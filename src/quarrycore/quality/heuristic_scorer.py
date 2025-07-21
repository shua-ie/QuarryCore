from __future__ import annotations

import asyncio
import re
from typing import TYPE_CHECKING, Dict, List

import structlog

from quarrycore.protocols import ContentMetadata, DomainType, ExtractedContent, QualityScore
from quarrycore.quality.scorer import Scorer

logger = structlog.get_logger(__name__)

if TYPE_CHECKING:
    from spacy.language import Language

try:
    import spacy

    nlp = spacy.load("en_core_web_sm")
except ImportError:
    logger.info("Downloading spacy model 'en_core_web_sm'...")
    import subprocess

    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    import spacy

    nlp = spacy.load("en_core_web_sm")


class HeuristicScorer:
    """
    Scores content based on heuristics like spam detection, information density,
    and domain relevance.
    """

    def __init__(self, domain_keywords: Dict[DomainType, List[str]] | None = None):
        """Initializes the HeuristicScorer."""
        try:
            self.nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
        except OSError:
            logger.info("Downloading spacy model 'en_core_web_sm'...")
            spacy.cli.download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])

        self.domain_keywords = domain_keywords or {}
        self.spam_keywords = [
            "free",
            "win",
            "winner",
            "cash",
            "prize",
            "limited time",
            "offer",
            "subscribe",
            "buy now",
            "click here",
            "urgent",
        ]

    async def score(
        self,
        content: ExtractedContent,
        metadata: ContentMetadata,
        score: QualityScore,
    ) -> None:
        """Calculates heuristic-based scores and updates the QualityScore object."""
        if not content.text or content.word_count < 10:
            return

        await asyncio.to_thread(self._calculate_scores, content, metadata, score)

    def _calculate_scores(
        self,
        content: ExtractedContent,
        metadata: ContentMetadata,
        score: QualityScore,
    ) -> None:
        """Synchronous method to calculate all heuristic scores."""
        text = content.text
        doc = self.nlp(text[: self.nlp.max_length])

        # 1. Information Density (entities per sentence)
        num_entities = len(doc.ents)
        num_sents = content.sentence_count or len(list(doc.sents))
        if num_sents > 0:
            info_density = num_entities / num_sents
        else:
            info_density = 0.0

        score.information_density = info_density
        score.quality_factors["information_density"] = info_density

        # 2. Domain Relevance (keyword matching)
        domain_relevance = 0.0
        if metadata.domain_type in self.domain_keywords:
            keywords = self.domain_keywords[metadata.domain_type]
            text_lower = text.lower()
            keyword_counts = sum(1 for k in keywords if k in text_lower)
            # Normalize by number of keywords for the domain
            domain_relevance = keyword_counts / len(keywords)

        score.domain_relevance = domain_relevance
        score.quality_factors["domain_relevance"] = domain_relevance

        # 3. Spam Score (heuristics)
        spam_score = self._calculate_spam_score(text, content.word_count)
        score.quality_factors["spam_score"] = spam_score

    def _calculate_spam_score(self, text: str, word_count: int) -> float:
        """Calculates a spam score based on multiple heuristics."""
        if word_count == 0:
            return 0.0

        scores = []

        # Keyword-based
        spam_keyword_count = sum(1 for k in self.spam_keywords if k in text.lower())
        scores.append(min(spam_keyword_count / 5.0, 1.0))  # Penalize up to 5 spam words

        # Uppercase ratio
        uppercase_chars = sum(1 for char in text if char.isupper())
        total_chars = len(text)
        if total_chars > 0:
            upper_ratio = uppercase_chars / total_chars
            # High uppercase ratio is a bad sign (e.g., > 30%)
            scores.append(min(upper_ratio / 0.3, 1.0))

        # Exclamation mark ratio
        exclamation_count = text.count("!")
        # High ratio is bad (e.g., > 5% of sentences end with !)
        scores.append(min((exclamation_count / (text.count(".") + 1)) / 0.1, 1.0))

        # Average these scores
        return sum(scores) / len(scores) if scores else 0.0
