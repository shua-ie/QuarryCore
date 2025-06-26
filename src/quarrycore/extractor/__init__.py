"""
QuarryCore Content Extraction Module - Multi-Strategy Cascade Extractor

This module provides sophisticated content extraction with a 4-tier cascade approach:
1. Primary: Trafilatura with precision optimization (90% F1 score)
2. Secondary: selectolax with Lexbor backend (70x faster than BeautifulSoup)  
3. Tertiary: LLM-assisted extraction for complex layouts
4. Fallback: Heuristic DOM-based extraction

Features:
- Multi-modal content extraction (text, tables, code, images, links)
- Domain-specific processing rules (medical, legal, e-commerce, technical)
- Language detection with fastText (50+ languages)
- Encoding normalization with charset-normalizer
- Extraction confidence scoring and quality assessment
- Hardware-adaptive processing for Pi to workstation scaling
"""

from .cascade_extractor import CascadeExtractor
from .content_processors import (
    TextProcessor,
    TableProcessor, 
    CodeProcessor,
    ImageProcessor,
    LinkProcessor,
)
from .domain_extractors import (
    MedicalExtractor,
    LegalExtractor,
    EcommerceExtractor,
    TechnicalExtractor,
)
from .language_detector import LanguageDetector
from .confidence_scorer import ConfidenceScorer

__all__ = [
    "CascadeExtractor",
    "TextProcessor",
    "TableProcessor",
    "CodeProcessor", 
    "ImageProcessor",
    "LinkProcessor",
    "MedicalExtractor",
    "LegalExtractor",
    "EcommerceExtractor",
    "TechnicalExtractor",
    "LanguageDetector",
    "ConfidenceScorer",
] 