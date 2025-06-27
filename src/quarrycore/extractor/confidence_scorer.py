"""
Extraction Confidence Scorer

Evaluates the quality and confidence of content extraction using
multiple signals including content characteristics, extraction method
performance, and domain-specific quality indicators.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional
from urllib.parse import urlparse

from ..protocols import CrawlResult, ExtractedContent

logger = logging.getLogger(__name__)


class ConfidenceScorer:
    """
    Multi-signal confidence scorer for extraction quality assessment.

    Evaluates extraction confidence based on:
    - Content length and structure
    - Text quality indicators
    - Extraction method reliability
    - Domain-specific quality factors
    - HTML structure analysis
    """

    def __init__(self) -> None:
        # Quality thresholds and weights
        self.quality_weights = {
            "content_length": 0.15,
            "structure_quality": 0.20,
            "text_quality": 0.25,
            "extraction_method": 0.15,
            "html_quality": 0.15,
            "domain_specific": 0.10,
        }

        # Extraction method reliability scores
        self.method_reliability = {
            "trafilatura": 0.95,
            "selectolax": 0.85,
            "llm_assisted": 0.90,
            "heuristic_fallback": 0.60,
            "cascade_trafilatura": 0.95,
            "cascade_selectolax": 0.85,
            "cascade_llm_assisted": 0.90,
            "cascade_heuristic_fallback": 0.60,
        }

        # Quality indicators
        self.quality_indicators = {
            "good_patterns": [
                r"\b(article|paragraph|section|content)\b",
                r"\b(introduction|conclusion|summary)\b",
                r"\b(first|second|third|finally|however|therefore)\b",
            ],
            "bad_patterns": [
                r"\b(click here|read more|advertisement|sponsored)\b",
                r"\b(cookie|privacy|terms|conditions)\b",
                r"\b(loading|error|404|not found)\b",
            ],
            "navigation_patterns": [
                r"\b(home|about|contact|menu|navigation)\b",
                r"\b(previous|next|back|forward)\b",
                r"\b(login|register|sign up|sign in)\b",
            ],
        }

        logger.info("ConfidenceScorer initialized")

    async def calculate_confidence(
        self,
        extracted_content: ExtractedContent,
        original_html: str,
        crawl_result: Optional[CrawlResult] = None,
    ) -> float:
        """
        Calculate overall extraction confidence score.

        Args:
            extracted_content: Extracted content to evaluate
            original_html: Original HTML content
            crawl_result: Optional crawl result for additional context

        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not extracted_content or not extracted_content.text:
            return 0.0

        scores = {}

        # 1. Content length score
        scores["content_length"] = await self._score_content_length(extracted_content)

        # 2. Structure quality score
        scores["structure_quality"] = await self._score_structure_quality(extracted_content)

        # 3. Text quality score
        scores["text_quality"] = await self._score_text_quality(extracted_content.text)

        # 4. Extraction method score
        scores["extraction_method"] = await self._score_extraction_method(extracted_content)

        # 5. HTML quality score
        scores["html_quality"] = await self._score_html_quality(extracted_content.text, original_html)

        # 6. Domain-specific score
        scores["domain_specific"] = await self._score_domain_specific(extracted_content, crawl_result)

        # Calculate weighted average
        total_score = 0.0
        for component, score in scores.items():
            weight = self.quality_weights.get(component, 0.0)
            total_score += score * weight

        # Apply penalties for extraction errors
        if extracted_content.extraction_errors:
            error_penalty = min(0.3, len(extracted_content.extraction_errors) * 0.1)
            total_score *= 1.0 - error_penalty

        # Normalize to 0-1 range
        final_score = max(0.0, min(1.0, total_score))

        logger.debug(f"Confidence scores: {scores}, final: {final_score:.3f}")

        return final_score

    async def _score_content_length(self, content: ExtractedContent) -> float:
        """Score based on content length appropriateness."""
        text_length = len(content.text)

        # Optimal range: 500-5000 characters
        if text_length < 50:
            return 0.0  # Too short
        elif text_length < 200:
            return 0.3  # Very short
        elif text_length < 500:
            return 0.6  # Short but acceptable
        elif text_length <= 5000:
            return 1.0  # Optimal range
        elif text_length <= 10000:
            return 0.9  # Long but good
        elif text_length <= 20000:
            return 0.7  # Very long
        else:
            return 0.5  # Extremely long, might be noisy

    async def _score_structure_quality(self, content: ExtractedContent) -> float:
        """Score based on content structure quality."""
        score = 0.0

        # Check word count
        if content.word_count > 50:
            score += 0.3

        # Check sentence count
        if content.sentence_count > 3:
            score += 0.2

        # Check paragraph count
        if content.paragraph_count > 1:
            score += 0.2

        # Check lexical diversity
        if hasattr(content, "lexical_diversity") and content.lexical_diversity > 0.3:
            score += 0.3

        return min(1.0, score)

    async def _score_text_quality(self, text: str) -> float:
        """Score based on text quality indicators."""
        if not text:
            return 0.0

        text_lower = text.lower()
        score = 0.5  # Base score

        # Check for good quality patterns
        good_matches = 0
        for pattern in self.quality_indicators["good_patterns"]:
            good_matches += len(re.findall(pattern, text_lower))

        # Check for bad quality patterns (navigation, ads, etc.)
        bad_matches = 0
        for pattern in self.quality_indicators["bad_patterns"]:
            bad_matches += len(re.findall(pattern, text_lower))

        # Check for navigation patterns
        nav_matches = 0
        for pattern in self.quality_indicators["navigation_patterns"]:
            nav_matches += len(re.findall(pattern, text_lower))

        # Calculate quality score
        text_length = len(text)
        if text_length > 0:
            # Normalize by text length
            good_density = (good_matches * 100) / text_length
            bad_density = (bad_matches * 100) / text_length
            nav_density = (nav_matches * 100) / text_length

            # Boost for good patterns
            score += min(0.3, good_density * 10)

            # Penalty for bad patterns
            score -= min(0.4, bad_density * 20)

            # Penalty for navigation patterns
            score -= min(0.2, nav_density * 15)

        # Check sentence structure
        sentences = re.split(r"[.!?]+", text)
        valid_sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

        if len(valid_sentences) > 0:
            avg_sentence_length = len(text) / len(valid_sentences)
            # Optimal sentence length: 50-150 characters
            if 50 <= avg_sentence_length <= 150:
                score += 0.1

        # Check for excessive repetition
        words = text_lower.split()
        if len(words) > 20:
            unique_words = set(words)
            repetition_ratio = len(unique_words) / len(words)
            if repetition_ratio < 0.3:  # High repetition
                score -= 0.2

        return max(0.0, min(1.0, score))

    async def _score_extraction_method(self, content: ExtractedContent) -> float:
        """Score based on extraction method reliability."""
        method = content.extraction_method.lower()

        # Look for method in reliability mapping
        for known_method, reliability in self.method_reliability.items():
            if known_method in method:
                return reliability

        # Default score for unknown methods
        return 0.5

    async def _score_html_quality(self, extracted_text: str, original_html: str) -> float:
        """Score based on HTML structure and extraction efficiency."""
        if not original_html:
            return 0.5

        score = 0.5  # Base score

        # Calculate extraction ratio
        html_text_content = re.sub(r"<[^>]+>", " ", original_html)
        html_text_content = re.sub(r"\s+", " ", html_text_content).strip()

        if len(html_text_content) > 0:
            extraction_ratio = len(extracted_text) / len(html_text_content)

            # Optimal extraction ratio: 20-80% of HTML text content
            if 0.2 <= extraction_ratio <= 0.8:
                score += 0.3
            elif 0.1 <= extraction_ratio < 0.2:
                score += 0.1
            elif 0.8 < extraction_ratio <= 1.0:
                score += 0.2
            else:
                score -= 0.1

        # Check HTML structure quality
        html_lower = original_html.lower()

        # Look for semantic HTML elements
        semantic_elements = [
            "article",
            "section",
            "main",
            "header",
            "footer",
            "nav",
            "aside",
        ]
        semantic_count = sum(html_lower.count(f"<{element}") for element in semantic_elements)

        if semantic_count > 0:
            score += min(0.2, semantic_count * 0.05)

        # Check for content indicators
        content_indicators = [
            'class="content"',
            'class="article"',
            'class="post"',
            'id="content"',
        ]
        if any(indicator in html_lower for indicator in content_indicators):
            score += 0.1

        # Penalty for excessive script/style content
        script_style_ratio = (html_lower.count("<script") + html_lower.count("<style")) / max(1, html_lower.count("<"))

        if script_style_ratio > 0.1:
            score -= min(0.2, script_style_ratio)

        return max(0.0, min(1.0, score))

    async def _score_domain_specific(
        self,
        content: ExtractedContent,
        crawl_result: Optional[CrawlResult],
    ) -> float:
        """Score based on domain-specific quality factors."""
        if not crawl_result or not crawl_result.url:
            return 0.5

        score = 0.5  # Base score
        text = content.text.lower()
        domain = urlparse(crawl_result.url).netloc.lower()

        # Domain reputation boost
        trusted_domains = [
            "wikipedia.org",
            "britannica.com",
            "nature.com",
            "sciencedirect.com",
            "pubmed.ncbi.nlm.nih.gov",
            "arxiv.org",
            "jstor.org",
            "springer.com",
            "gov",
            "edu",
            "ac.uk",
            "edu.au",
        ]

        if any(trusted in domain for trusted in trusted_domains):
            score += 0.3

        # Domain-specific content quality
        if "wikipedia.org" in domain:
            # Wikipedia-specific quality checks
            if "references" in text or "citation" in text:
                score += 0.1
            if len(content.text) > 1000:  # Wikipedia articles should be substantial
                score += 0.1

        elif any(academic in domain for academic in ["edu", "ac.", "arxiv", "pubmed"]):
            # Academic content quality
            academic_terms = [
                "research",
                "study",
                "analysis",
                "methodology",
                "conclusion",
            ]
            academic_count = sum(text.count(term) for term in academic_terms)
            if academic_count > 5:
                score += 0.2

        elif "gov" in domain:
            # Government content quality
            official_terms = ["official", "regulation", "policy", "public", "federal"]
            official_count = sum(text.count(term) for term in official_terms)
            if official_count > 3:
                score += 0.2

        # News domain quality
        news_domains = ["bbc.", "cnn.", "reuters.", "ap.org", "npr.org", "nytimes."]
        if any(news in domain for news in news_domains):
            # Check for news article structure
            if any(indicator in text for indicator in ["reported", "according to", "sources say"]):
                score += 0.1

        # Check for spam indicators
        spam_indicators = [
            "click here",
            "buy now",
            "free trial",
            "limited time",
            "act now",
            "call now",
            "subscribe",
            "sign up now",
        ]
        spam_count = sum(text.count(indicator) for indicator in spam_indicators)
        if spam_count > 3:
            score -= 0.3

        return max(0.0, min(1.0, score))

    async def get_confidence_breakdown(
        self,
        extracted_content: ExtractedContent,
        original_html: str,
        crawl_result: Optional[CrawlResult] = None,
    ) -> Dict[str, float]:
        """
        Get detailed breakdown of confidence scoring components.

        Returns:
            Dictionary with individual component scores
        """
        breakdown = {}

        breakdown["content_length"] = await self._score_content_length(extracted_content)
        breakdown["structure_quality"] = await self._score_structure_quality(extracted_content)
        breakdown["text_quality"] = await self._score_text_quality(extracted_content.text)
        breakdown["extraction_method"] = await self._score_extraction_method(extracted_content)
        breakdown["html_quality"] = await self._score_html_quality(extracted_content.text, original_html)
        breakdown["domain_specific"] = await self._score_domain_specific(extracted_content, crawl_result)

        # Add weighted scores
        breakdown["weighted_scores"] = {
            component: score * self.quality_weights.get(component, 0.0)
            for component, score in breakdown.items()
            if component != "weighted_scores"
        }

        return breakdown

    def update_quality_weights(self, new_weights: Dict[str, float]) -> None:
        """Update quality component weights."""
        # Normalize weights to sum to 1.0
        total_weight = sum(new_weights.values())
        if total_weight > 0:
            self.quality_weights.update({component: weight / total_weight for component, weight in new_weights.items()})
            logger.info(f"Updated quality weights: {self.quality_weights}")

    def update_method_reliability(self, method: str, reliability: float) -> None:
        """Update reliability score for extraction method."""
        self.method_reliability[method.lower()] = max(0.0, min(1.0, reliability))
        logger.info(f"Updated {method} reliability to {reliability}")

    async def benchmark_extraction_quality(
        self,
        extractions: List[tuple[ExtractedContent, str, CrawlResult]],  # (ExtractedContent, original_html, CrawlResult)
    ) -> Dict[str, float]:
        """
        Benchmark extraction quality across multiple samples.

        Args:
            extractions: List of (content, html, crawl_result) tuples

        Returns:
            Aggregated quality statistics
        """
        if not extractions:
            return {}

        all_scores = []
        component_scores: Dict[str, List[float]] = {component: [] for component in self.quality_weights.keys()}

        for content, html, crawl_result in extractions:
            overall_score = await self.calculate_confidence(content, html, crawl_result)
            all_scores.append(overall_score)

            breakdown = await self.get_confidence_breakdown(content, html, crawl_result)
            for component, score in breakdown.items():
                if component in component_scores:
                    component_scores[component].append(score)

        # Calculate statistics
        stats = {
            "count": len(all_scores),
            "mean_confidence": sum(all_scores) / len(all_scores),
            "min_confidence": min(all_scores),
            "max_confidence": max(all_scores),
        }

        # Component averages
        component_averages: Dict[str, float] = {
            component: sum(scores) / len(scores) if scores else 0.0 for component, scores in component_scores.items()
        }
        stats["component_averages"] = component_averages

        return stats
