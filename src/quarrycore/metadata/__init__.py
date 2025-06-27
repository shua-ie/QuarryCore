"""
QuarryCore Metadata Extraction Module - Comprehensive Content Analysis

This module provides sophisticated metadata extraction and content analysis capabilities:

Core Features:
- OpenGraph and Schema.org structured data parsing
- Author entity recognition with spaCy NER
- Multi-strategy publication date detection
- Social media metrics extraction
- Content categorization and quality scoring
- Reading time and lexical diversity analysis
- DOM complexity and structure analysis
- Domain-specific metadata enrichment

Components:
- MetadataExtractor: Main extraction coordinator
- StructuredDataParser: OpenGraph/Schema.org parsing
- AuthorExtractor: Author identification and entity recognition
- DateExtractor: Multi-strategy publication date detection
- SocialMetricsExtractor: Social media engagement metrics
- ContentAnalyzer: Quality indicators and categorization
- DOMAnalyzer: HTML structure and complexity analysis
- QualityScorer: Comprehensive content quality assessment

Performance Targets:
- Raspberry Pi: Fast metadata extraction with core features
- Workstation: Full analysis with ML-based categorization
- Enterprise: Batch processing with domain-specific enrichment
"""

from .author_extractor import AuthorConfidence, AuthorExtractor, AuthorInfo
from .content_analyzer import ContentAnalyzer, ContentMetrics, LexicalMetrics, QualityIndicators, ReadingMetrics
from .date_extractor import DateExtractionStrategy, DateExtractor, DateInfo
from .dom_analyzer import DOMAnalyzer, DOMMetrics, StructureMetrics
from .metadata_extractor import MetadataExtractor
from .quality_scorer import QualityFactors, QualityScore, QualityScorer
from .social_metrics_extractor import PlatformMetrics, SocialMetrics, SocialMetricsExtractor
from .structured_data_parser import OpenGraphParser, SchemaOrgParser, StructuredDataParser, TwitterCardParser

__all__ = [
    # Main extractor
    "MetadataExtractor",
    # Structured data parsing
    "StructuredDataParser",
    "OpenGraphParser",
    "SchemaOrgParser",
    "TwitterCardParser",
    # Author extraction
    "AuthorExtractor",
    "AuthorInfo",
    "AuthorConfidence",
    # Date extraction
    "DateExtractor",
    "DateInfo",
    "DateExtractionStrategy",
    # Social metrics
    "SocialMetricsExtractor",
    "SocialMetrics",
    "PlatformMetrics",
    # Content analysis
    "ContentAnalyzer",
    "ContentMetrics",
    "QualityIndicators",
    "ReadingMetrics",
    "LexicalMetrics",
    # DOM analysis
    "DOMAnalyzer",
    "DOMMetrics",
    "StructureMetrics",
    # Quality scoring
    "QualityScorer",
    "QualityScore",
    "QualityFactors",
]
