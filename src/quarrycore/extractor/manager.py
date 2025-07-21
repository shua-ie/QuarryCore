"""
Production-grade ExtractorManager for QuarryCore.

Orchestrates multiple extractors in a configurable cascade with quality gating.
"""

from __future__ import annotations

import asyncio
import time
from typing import Dict, List, Optional, Protocol, Sequence

import structlog

from ..config.config import ExtractionSettings
from ..protocols import QualityProtocol
from .models import ExtractResult
from .protocols import Extractor
from .readability_extractor import ReadabilityExtractor
from .soup_extractor import SoupFallbackExtractor
from .trafilatura_extractor import TrafilaturaExtractor

logger = structlog.get_logger(__name__)


class ExtractorManager:
    """
    Manages cascading extraction strategies with quality assessment.

    Features:
    - Configurable extractor cascade order
    - Domain-specific extractor ordering
    - Quality threshold filtering
    - Resilient error handling with fallback
    - Performance metrics tracking
    """

    def __init__(
        self,
        quality_assessor: QualityProtocol,
        settings: ExtractionSettings,
    ) -> None:
        """
        Initialize the ExtractorManager.

        Args:
            quality_assessor: Quality assessment service
            settings: Extraction configuration settings
        """
        self.quality_assessor = quality_assessor
        self.settings = settings
        self.logger = logger.bind(component="ExtractorManager")

        # Initialize available extractors
        self._extractors: Dict[str, Extractor] = {
            "trafilatura": TrafilaturaExtractor(),
            "readability": ReadabilityExtractor(),
            "soup_fallback": SoupFallbackExtractor(),
        }

        # Validate cascade order
        self._validate_cascade_order()

        # Performance metrics
        self._extraction_metrics: Dict[str, Dict[str, float]] = {
            extractor_name: {"attempts": 0, "successes": 0, "total_time": 0.0} for extractor_name in self._extractors
        }

    def _validate_cascade_order(self) -> None:
        """Validate that cascade order contains valid extractor names."""
        for extractor_name in self.settings.cascade_order:
            if extractor_name not in self._extractors:
                raise ValueError(
                    f"Invalid extractor '{extractor_name}' in cascade_order. "
                    f"Available extractors: {list(self._extractors.keys())}"
                )

        # Validate domain overrides
        for domain, order in self.settings.domain_overrides.items():
            for extractor_name in order:
                if extractor_name not in self._extractors:
                    raise ValueError(
                        f"Invalid extractor '{extractor_name}' in domain override for '{domain}'. "
                        f"Available extractors: {list(self._extractors.keys())}"
                    )

    def _get_cascade_order(self, url: str) -> List[str]:
        """
        Determine extractor cascade order for a given URL.

        Args:
            url: The URL being extracted

        Returns:
            List of extractor names in cascade order
        """
        # Extract domain from URL
        from urllib.parse import urlparse

        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()

            # Check for exact domain match
            if domain in self.settings.domain_overrides:
                return self.settings.domain_overrides[domain]

            # Check for subdomain matches (e.g., "nytimes.com" matches "www.nytimes.com")
            for override_domain, override_order in self.settings.domain_overrides.items():
                if domain.endswith(f".{override_domain}") or domain == override_domain:
                    return override_order

        except Exception as e:
            self.logger.warning("Failed to parse domain for override check", url=url, error=str(e))

        # Fall back to global cascade order
        return self.settings.cascade_order

    async def extract(self, url: str, html: str) -> Optional[ExtractResult]:
        """
        Extract content using cascading strategies with quality gating.

        Args:
            url: Source URL
            html: HTML content to extract from

        Returns:
            ExtractResult if quality threshold met, None otherwise
        """
        cascade_order = self._get_cascade_order(url)

        self.logger.info(
            "Starting extraction cascade",
            url=url,
            cascade_order=cascade_order,
            quality_threshold=self.settings.quality_threshold,
        )

        for extractor_name in cascade_order:
            extractor = self._extractors[extractor_name]

            # Track metrics
            start_time = time.time()
            self._extraction_metrics[extractor_name]["attempts"] += 1

            try:
                # Attempt extraction
                self.logger.debug("Attempting extraction", extractor=extractor_name, url=url)

                result = await extractor.extract(html, url=url)

                # Record extraction time
                extraction_time = time.time() - start_time
                self._extraction_metrics[extractor_name]["total_time"] += extraction_time

                # Skip if no content extracted
                if not result.text.strip():
                    self.logger.debug("Extractor returned empty content", extractor=extractor_name, url=url)
                    continue

                # Assess quality
                quality_start = time.time()
                # Create minimal metadata for quality assessment
                from ..protocols import ContentMetadata, DomainType

                metadata = ContentMetadata(url=url, domain_type=DomainType.GENERAL)
                from ..protocols import ExtractedContent as ProtocolExtractedContent

                extracted_content = ProtocolExtractedContent(text=result.text, extraction_method=extractor_name)
                quality_result = await self.quality_assessor.assess_quality(
                    content=extracted_content, metadata=metadata
                )
                quality_score = quality_result.overall_score
                quality_time = time.time() - quality_start

                # Update result with quality score
                result = ExtractResult(
                    url=result.url,
                    text=result.text,
                    title=result.title,
                    images=result.images,
                    language=result.language,
                    score=quality_score,
                )

                self.logger.info(
                    "Extraction completed",
                    extractor=extractor_name,
                    url=url,
                    quality_score=quality_score,
                    extraction_time=extraction_time,
                    quality_time=quality_time,
                    text_length=len(result.text),
                    passed_threshold=quality_score >= self.settings.quality_threshold,
                )

                # Check quality threshold
                if quality_score >= self.settings.quality_threshold:
                    self._extraction_metrics[extractor_name]["successes"] += 1
                    return result
                else:
                    self.logger.debug(
                        "Content below quality threshold",
                        extractor=extractor_name,
                        url=url,
                        quality_score=quality_score,
                        threshold=self.settings.quality_threshold,
                    )
                    continue

            except Exception as e:
                # Log error and continue to next extractor
                self.logger.error(
                    "Extractor failed",
                    event_type="extractor_failed",
                    extractor=extractor_name,
                    url=url,
                    error=str(e),
                    error_type=type(e).__name__,
                )
                continue

        # All extractors failed or returned low-quality content
        self.logger.warning(
            "All extractors failed or produced low-quality content",
            url=url,
            cascade_order=cascade_order,
            threshold=self.settings.quality_threshold,
        )

        return None

    def get_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Get extraction performance metrics.

        Returns:
            Dictionary of metrics per extractor
        """
        metrics = {}

        for extractor_name, raw_metrics in self._extraction_metrics.items():
            attempts = raw_metrics["attempts"]
            successes = raw_metrics["successes"]
            total_time = raw_metrics["total_time"]

            metrics[extractor_name] = {
                "attempts": attempts,
                "successes": successes,
                "success_rate": successes / attempts if attempts > 0 else 0.0,
                "total_time": total_time,
                "avg_time": total_time / attempts if attempts > 0 else 0.0,
            }

        return metrics
