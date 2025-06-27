"""
Main Metadata Extractor - Comprehensive Content Analysis Coordinator

Orchestrates all metadata extraction components to provide comprehensive content analysis
for AI training data mining with domain-specific enrichment.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, TypedDict

from ..protocols import ContentMetadata, CrawlResult, DomainType, ExtractedContent, HardwareCapabilities
from .author_extractor import AuthorExtractor
from .content_analyzer import ContentAnalyzer
from .date_extractor import DateExtractor
from .dom_analyzer import DOMAnalyzer
from .quality_scorer import QualityScorer
from .social_metrics_extractor import SocialMetricsExtractor

# Import component extractors
from .structured_data_parser import StructuredDataParser

logger = logging.getLogger(__name__)


class ComponentStats(TypedDict):
    """Type definition for component performance statistics."""

    calls: int
    avg_time: float
    success_rate: float


class ExtractionStats(TypedDict):
    """Type definition for overall extraction statistics."""

    total_extractions: int
    successful_extractions: int
    avg_processing_time: float
    component_performance: Dict[str, ComponentStats]


@dataclass
class MetadataExtractionConfig:
    """Configuration for metadata extraction process."""

    # Core extraction features
    extract_structured_data: bool = True
    extract_authors: bool = True
    extract_dates: bool = True
    extract_social_metrics: bool = True
    analyze_content_quality: bool = True
    analyze_dom_structure: bool = True

    # Advanced features
    use_nlp_models: bool = True
    enable_entity_linking: bool = True
    calculate_reading_metrics: bool = True
    analyze_lexical_diversity: bool = True

    # Domain-specific processing
    enable_domain_enrichment: bool = True
    domain_confidence_threshold: float = 0.7

    # Performance settings
    parallel_processing: bool = True
    cache_nlp_models: bool = True
    timeout_seconds: float = 30.0

    # Quality thresholds
    min_content_length: int = 100
    min_quality_score: float = 0.3
    require_publication_date: bool = False
    require_author_info: bool = False


class MetadataExtractor:
    """
    Comprehensive metadata extractor for content analysis.

    Coordinates multiple extraction strategies to provide rich metadata
    for AI training data including structured data, authorship, temporal
    information, social engagement, and quality indicators.
    """

    def __init__(
        self,
        config: Optional[MetadataExtractionConfig] = None,
        hardware_caps: Optional[HardwareCapabilities] = None,
    ):
        self.config = config or MetadataExtractionConfig()
        self.hardware_caps = hardware_caps

        # Initialize component extractors
        self.structured_parser = StructuredDataParser()
        self.author_extractor = AuthorExtractor()
        self.date_extractor = DateExtractor()
        self.social_extractor = SocialMetricsExtractor()
        self.content_analyzer = ContentAnalyzer()
        self.dom_analyzer = DOMAnalyzer()
        self.quality_scorer = QualityScorer()

        # Performance tracking with proper typing
        self._extraction_stats: ExtractionStats = {
            "total_extractions": 0,
            "successful_extractions": 0,
            "avg_processing_time": 0.0,
            "component_performance": {
                "structured_data": {"calls": 0, "avg_time": 0.0, "success_rate": 0.0},
                "author_extraction": {"calls": 0, "avg_time": 0.0, "success_rate": 0.0},
                "date_extraction": {"calls": 0, "avg_time": 0.0, "success_rate": 0.0},
                "social_metrics": {"calls": 0, "avg_time": 0.0, "success_rate": 0.0},
                "content_analysis": {"calls": 0, "avg_time": 0.0, "success_rate": 0.0},
                "dom_analysis": {"calls": 0, "avg_time": 0.0, "success_rate": 0.0},
                "quality_scoring": {"calls": 0, "avg_time": 0.0, "success_rate": 0.0},
            },
        }

        # Adapt to hardware if provided
        if hardware_caps:
            self._adapt_to_hardware(hardware_caps)

        logger.info("MetadataExtractor initialized with comprehensive analysis")

    def _adapt_to_hardware(self, capabilities: HardwareCapabilities) -> None:
        """Adapt extraction configuration to hardware capabilities."""
        # Disable resource-intensive features on Raspberry Pi
        if capabilities.hardware_type.value == "raspberry_pi":
            self.config.use_nlp_models = False
            self.config.enable_entity_linking = False
            self.config.parallel_processing = False
            self.config.timeout_seconds = 15.0
            logger.info("Adapted metadata extractor for Raspberry Pi: disabled NLP models")

        # Enable all features on workstations
        elif capabilities.hardware_type.value == "workstation":
            self.config.use_nlp_models = True
            self.config.enable_entity_linking = True
            self.config.parallel_processing = True
            self.config.cache_nlp_models = True
            logger.info("Adapted metadata extractor for workstation: enabled all features")

        # Adjust timeout based on CPU performance
        base_timeout = 30.0
        cpu_factor = max(0.5, capabilities.cpu_cores / 8.0)
        self.config.timeout_seconds = base_timeout / cpu_factor

    async def extract_metadata(
        self,
        crawl_result: CrawlResult,
        extracted_content: Optional[ExtractedContent] = None,
        *,
        domain_type: Optional[DomainType] = None,
    ) -> ContentMetadata:
        """
        Extract comprehensive metadata from crawl result and content.

        Args:
            crawl_result: Web crawling result with HTML content
            extracted_content: Previously extracted content (optional)
            domain_type: Detected domain type for specialized processing

        Returns:
            ContentMetadata with comprehensive analysis results
        """
        start_time = time.time()

        # Initialize metadata result
        metadata = ContentMetadata(
            url=crawl_result.url,
        )

        try:
            # Validate input
            if not crawl_result.content or not crawl_result.is_valid:
                return metadata

            # Decode HTML content
            try:
                html_content = crawl_result.content.decode("utf-8", errors="replace")
            except Exception:
                return metadata

            # Check content length requirements
            if len(html_content) < self.config.min_content_length:
                pass

            # Extract text content if not provided
            text_content = ""
            if extracted_content and extracted_content.text:
                text_content = extracted_content.text
            else:
                # Simple text extraction for metadata analysis
                text_content = await self._extract_text_for_analysis(html_content)

            # Run extraction components in parallel if enabled
            if self.config.parallel_processing:
                await self._extract_parallel(metadata, html_content, text_content, crawl_result)
            else:
                await self._extract_sequential(metadata, html_content, text_content, crawl_result)

            # Apply domain-specific enrichment
            if self.config.enable_domain_enrichment and domain_type:
                await self._apply_domain_enrichment(metadata, domain_type, html_content, text_content)

            # Calculate overall quality score
            if self.config.analyze_content_quality:
                quality_score = await self.quality_scorer.calculate_overall_quality(
                    metadata, html_content, text_content
                )
                metadata.quality_score = quality_score.overall_score

            # Validate quality requirements
            if metadata.quality_score < self.config.min_quality_score:
                pass

            # Check required fields
            if self.config.require_publication_date and not metadata.publication_date:
                pass

            if self.config.require_author_info and not metadata.authors:
                pass

            # Update success statistics
            self._extraction_stats["successful_extractions"] += 1

        except Exception as e:
            logger.error(f"Metadata extraction error for {crawl_result.url}: {e}")
            metadata.errors.append(f"Unexpected error: {str(e)}")

        finally:
            # Record performance metrics
            total_time = (time.time() - start_time) * 1000
            self._extraction_stats["total_extractions"] += 1
            self._update_performance_stats(total_time)

        return metadata

    async def _extract_parallel(
        self,
        metadata: ContentMetadata,
        html_content: str,
        text_content: str,
        crawl_result: CrawlResult,
    ) -> None:
        """Run extraction components in parallel for better performance."""
        tasks = []

        # Structured data extraction
        if self.config.extract_structured_data:
            tasks.append(self._extract_structured_data(metadata, html_content))

        # Author extraction
        if self.config.extract_authors:
            tasks.append(self._extract_authors(metadata, html_content, text_content))

        # Date extraction
        if self.config.extract_dates:
            tasks.append(self._extract_dates(metadata, html_content, crawl_result.url))

        # Social metrics extraction
        if self.config.extract_social_metrics:
            tasks.append(self._extract_social_metrics(metadata, html_content))

        # Content analysis
        if self.config.analyze_content_quality:
            tasks.append(self._analyze_content(metadata, text_content, html_content))

        # DOM analysis
        if self.config.analyze_dom_structure:
            tasks.append(self._analyze_dom_structure(metadata, html_content))

        # Execute all tasks in parallel
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _extract_sequential(
        self,
        metadata: ContentMetadata,
        html_content: str,
        text_content: str,
        crawl_result: CrawlResult,
    ) -> None:
        """Run extraction components sequentially for resource-constrained environments."""
        # Structured data extraction
        if self.config.extract_structured_data:
            await self._extract_structured_data(metadata, html_content)

        # Author extraction
        if self.config.extract_authors:
            await self._extract_authors(metadata, html_content, text_content)

        # Date extraction
        if self.config.extract_dates:
            await self._extract_dates(metadata, html_content, crawl_result.url)

        # Social metrics extraction
        if self.config.extract_social_metrics:
            await self._extract_social_metrics(metadata, html_content)

        # Content analysis
        if self.config.analyze_content_quality:
            await self._analyze_content(metadata, text_content, html_content)

        # DOM analysis
        if self.config.analyze_dom_structure:
            await self._analyze_dom_structure(metadata, html_content)

    async def _extract_structured_data(self, metadata: ContentMetadata, html_content: str) -> None:
        """Extract structured data (OpenGraph, Schema.org, Twitter Cards)."""
        start_time = time.time()

        try:
            structured_data = await self.structured_parser.parse_all(html_content)

            # Extract key metadata from structured data
            if structured_data.get("og_title"):
                metadata.title = structured_data["og_title"]
            elif structured_data.get("schema_title"):
                metadata.title = structured_data["schema_title"]

            if structured_data.get("og_description"):
                metadata.description = structured_data["og_description"]
            elif structured_data.get("schema_description"):
                metadata.description = structured_data["schema_description"]

            if structured_data.get("og_image"):
                metadata.featured_image = structured_data["og_image"]

            # Store all structured data
            metadata.structured_data = structured_data

            # Update performance stats
            processing_time = (time.time() - start_time) * 1000
            self._update_component_stats("structured_data", processing_time, True)

        except Exception as e:
            logger.warning(f"Structured data extraction failed: {e}")
            self._update_component_stats("structured_data", (time.time() - start_time) * 1000, False)

    async def _extract_authors(self, metadata: ContentMetadata, html_content: str, text_content: str) -> None:
        """Extract author information using multiple strategies."""
        start_time = time.time()

        try:
            authors = await self.author_extractor.extract_authors(
                html_content, text_content, use_nlp=self.config.use_nlp_models
            )

            metadata.authors = authors

            # Update performance stats
            processing_time = (time.time() - start_time) * 1000
            self._update_component_stats("author_extraction", processing_time, bool(authors))

        except Exception as e:
            logger.warning(f"Author extraction failed: {e}")
            self._update_component_stats("author_extraction", (time.time() - start_time) * 1000, False)

    async def _extract_dates(self, metadata: ContentMetadata, html_content: str, url: str) -> None:
        """Extract publication date using multiple strategies."""
        start_time = time.time()

        try:
            date_info = await self.date_extractor.extract_publication_date(html_content, url)

            if date_info:
                metadata.publication_date = date_info.date
                metadata.date_confidence = date_info.confidence
                metadata.date_extraction_method = date_info.extraction_method.value

            # Update performance stats
            processing_time = (time.time() - start_time) * 1000
            self._update_component_stats("date_extraction", processing_time, bool(date_info))

        except Exception as e:
            logger.warning(f"Date extraction failed: {e}")
            self._update_component_stats("date_extraction", (time.time() - start_time) * 1000, False)

    async def _extract_social_metrics(self, metadata: ContentMetadata, html_content: str) -> None:
        """Extract social media metrics and engagement data."""
        start_time = time.time()

        try:
            social_metrics = await self.social_extractor.extract_metrics(html_content)

            if social_metrics:
                metadata.social_metrics = asdict(social_metrics)

            # Update performance stats
            processing_time = (time.time() - start_time) * 1000
            self._update_component_stats("social_metrics", processing_time, bool(social_metrics))

        except Exception as e:
            logger.warning(f"Social metrics extraction failed: {e}")
            self._update_component_stats("social_metrics", (time.time() - start_time) * 1000, False)

    async def _analyze_content(self, metadata: ContentMetadata, text_content: str, html_content: str) -> None:
        """Analyze content quality, reading metrics, and categorization."""
        start_time = time.time()

        try:
            content_metrics = await self.content_analyzer.analyze_content(
                text_content,
                html_content,
                calculate_reading_metrics=self.config.calculate_reading_metrics,
                analyze_lexical_diversity=self.config.analyze_lexical_diversity,
            )

            # Extract key metrics
            metadata.word_count = content_metrics.word_count
            metadata.reading_time_minutes = content_metrics.reading_time_minutes
            metadata.lexical_diversity = content_metrics.lexical_diversity
            metadata.content_categories = content_metrics.categories
            metadata.quality_indicators = asdict(content_metrics.quality_indicators)

            # Update performance stats
            processing_time = (time.time() - start_time) * 1000
            self._update_component_stats("content_analysis", processing_time, True)

        except Exception as e:
            logger.warning(f"Content analysis failed: {e}")
            self._update_component_stats("content_analysis", (time.time() - start_time) * 1000, False)

    async def _analyze_dom_structure(self, metadata: ContentMetadata, html_content: str) -> None:
        """Analyze DOM structure and complexity."""
        start_time = time.time()

        try:
            dom_metrics = await self.dom_analyzer.analyze_structure(html_content)

            metadata.dom_metrics = dom_metrics

            # Update performance stats
            processing_time = (time.time() - start_time) * 1000
            self._update_component_stats("dom_analysis", processing_time, True)

        except Exception as e:
            logger.warning(f"DOM analysis failed: {e}")
            self._update_component_stats("dom_analysis", (time.time() - start_time) * 1000, False)

    async def _apply_domain_enrichment(
        self,
        metadata: ContentMetadata,
        domain_type: DomainType,
        html_content: str,
        text_content: str,
    ) -> None:
        """Apply domain-specific metadata enrichment."""
        try:
            # Domain-specific processing based on type
            if domain_type == DomainType.MEDICAL:
                await self._enrich_medical_metadata(metadata, text_content)
            elif domain_type == DomainType.LEGAL:
                await self._enrich_legal_metadata(metadata, text_content)
            elif domain_type == DomainType.ECOMMERCE:
                await self._enrich_ecommerce_metadata(metadata, html_content)
            elif domain_type == DomainType.TECHNICAL:
                await self._enrich_technical_metadata(metadata, text_content)

        except Exception as e:
            logger.warning(f"Domain enrichment failed for {domain_type}: {e}")

    async def _enrich_medical_metadata(self, metadata: ContentMetadata, text_content: str) -> None:
        """Enrich metadata for medical content."""
        # Add medical-specific quality indicators
        medical_indicators = {
            "has_medical_terms": any(
                term in text_content.lower() for term in ["patient", "diagnosis", "treatment", "clinical", "medical"]
            ),
            "has_citations": "pubmed" in text_content.lower() or "doi:" in text_content.lower(),
            "has_disclaimers": any(
                term in text_content.lower() for term in ["disclaimer", "consult", "physician", "doctor"]
            ),
        }

        if not metadata.quality_indicators:
            metadata.quality_indicators = {}
        metadata.quality_indicators.update(medical_indicators)

    async def _enrich_legal_metadata(self, metadata: ContentMetadata, text_content: str) -> None:
        """Enrich metadata for legal content."""
        # Add legal-specific quality indicators
        legal_indicators = {
            "has_legal_citations": any(
                pattern in text_content for pattern in ["v.", "F.2d", "F.3d", "U.S.C.", "C.F.R."]
            ),
            "has_jurisdiction": any(term in text_content.lower() for term in ["court", "federal", "state", "district"]),
            "has_case_law": "precedent" in text_content.lower() or "ruling" in text_content.lower(),
        }

        if not metadata.quality_indicators:
            metadata.quality_indicators = {}
        metadata.quality_indicators.update(legal_indicators)

    async def _enrich_ecommerce_metadata(self, metadata: ContentMetadata, html_content: str) -> None:
        """Enrich metadata for e-commerce content."""
        # Add e-commerce specific quality indicators
        ecommerce_indicators = {
            "has_price": any(symbol in html_content for symbol in ["$", "€", "£", "¥"]),
            "has_reviews": any(term in html_content.lower() for term in ["review", "rating", "stars", "customer"]),
            "has_product_info": any(term in html_content.lower() for term in ["sku", "model", "brand", "warranty"]),
        }

        if not metadata.quality_indicators:
            metadata.quality_indicators = {}
        metadata.quality_indicators.update(ecommerce_indicators)

    async def _enrich_technical_metadata(self, metadata: ContentMetadata, text_content: str) -> None:
        """Enrich metadata for technical content."""
        # Add technical-specific quality indicators
        technical_indicators = {
            "has_code_examples": any(
                term in text_content.lower() for term in ["function", "class", "import", "def ", "var "]
            ),
            "has_api_docs": any(term in text_content.lower() for term in ["api", "endpoint", "request", "response"]),
            "has_version_info": any(term in text_content for term in ["v1.", "v2.", "version", "release"]),
        }

        if not metadata.quality_indicators:
            metadata.quality_indicators = {}
        metadata.quality_indicators.update(technical_indicators)

    async def _extract_text_for_analysis(self, html_content: str) -> str:
        """Simple text extraction for metadata analysis when content not provided."""
        try:
            # Use BeautifulSoup for text extraction
            try:
                from bs4 import BeautifulSoup

                soup = BeautifulSoup(html_content, "html.parser")

                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()

                # Get text content
                text = soup.get_text()

                # Clean up whitespace
                import re

                text = re.sub(r"\s+", " ", text).strip()

                return text

            except ImportError:
                # Fallback to simple regex if BeautifulSoup not available
                import re

                text = re.sub(r"<[^>]+>", " ", html_content)
                text = re.sub(r"\s+", " ", text).strip()
                return text

        except Exception as e:
            logger.warning(f"Text extraction failed: {e}")
            return ""

    def _update_component_stats(self, component: str, processing_time: float, success: bool) -> None:
        """Update performance statistics for extraction components."""
        stats = self._extraction_stats["component_performance"][component]

        # Update call count with proper typing
        stats["calls"] += 1

        # Update average processing time
        current_avg = stats["avg_time"]
        current_count = stats["calls"]
        new_avg = ((current_avg * (current_count - 1)) + processing_time) / current_count
        stats["avg_time"] = new_avg

        # Update success rate
        current_rate = stats["success_rate"]
        success_value = 1.0 if success else 0.0
        new_rate = ((current_rate * (current_count - 1)) + success_value) / current_count
        stats["success_rate"] = new_rate

    def _update_performance_stats(self, processing_time: float) -> None:
        """Update overall performance statistics."""
        current_avg = self._extraction_stats["avg_processing_time"]
        current_count = self._extraction_stats["total_extractions"]

        if current_count > 1:
            new_avg = ((current_avg * (current_count - 1)) + processing_time) / current_count
        else:
            new_avg = processing_time

        self._extraction_stats["avg_processing_time"] = new_avg

    async def extract_batch(
        self,
        crawl_results: List[CrawlResult],
        extracted_contents: Optional[List[ExtractedContent]] = None,
        *,
        parallel_workers: Optional[int] = None,
    ) -> List[ContentMetadata]:
        """Extract metadata from multiple crawl results in parallel."""
        # Determine parallelism
        max_workers = parallel_workers or (self.hardware_caps.recommended_workers if self.hardware_caps else 4)

        if not self.config.parallel_processing:
            max_workers = 1

        # Process in batches
        semaphore = asyncio.Semaphore(max_workers)

        async def extract_with_semaphore(i: int, crawl_result: CrawlResult) -> ContentMetadata:
            async with semaphore:
                extracted_content = extracted_contents[i] if extracted_contents else None
                return await self.extract_metadata(crawl_result, extracted_content)

        # Create tasks
        tasks = [extract_with_semaphore(i, result) for i, result in enumerate(crawl_results)]

        # Execute and return results
        metadata_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions with proper type safety
        final_results: List[ContentMetadata] = []
        for i, result in enumerate(metadata_results):
            if isinstance(result, Exception):
                # Create error result
                error_metadata = ContentMetadata(
                    url=crawl_results[i].url,
                )
                error_metadata.errors.append(f"Exception during extraction: {str(result)}")
                final_results.append(error_metadata)
            elif isinstance(result, ContentMetadata):
                final_results.append(result)

        return final_results

    def get_extraction_stats(self) -> Dict[str, Any]:
        """Get current extraction performance statistics."""
        # Convert TypedDict to regular dict for API compatibility
        return {
            "total_extractions": self._extraction_stats["total_extractions"],
            "successful_extractions": self._extraction_stats["successful_extractions"],
            "avg_processing_time": self._extraction_stats["avg_processing_time"],
            "component_performance": dict(self._extraction_stats["component_performance"]),
        }

    async def adapt_to_hardware(self, capabilities: HardwareCapabilities) -> None:
        """Adapt extractor settings based on hardware capabilities."""
        self.hardware_caps = capabilities
        self._adapt_to_hardware(capabilities)
