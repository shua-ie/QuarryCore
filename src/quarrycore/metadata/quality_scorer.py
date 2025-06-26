"""
Quality Scorer - Comprehensive Content Quality Assessment

Calculates overall quality scores by combining metrics from all metadata
extraction components for comprehensive content evaluation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class QualityFactors:
    """Individual quality factors contributing to overall score."""
    
    # Content quality factors (0-1 scale)
    content_length: float = 0.0
    readability: float = 0.0
    lexical_diversity: float = 0.0
    grammar_quality: float = 0.0
    
    # Structure quality factors
    html_structure: float = 0.0
    semantic_markup: float = 0.0
    accessibility: float = 0.0
    
    # Metadata quality factors
    metadata_completeness: float = 0.0
    author_credibility: float = 0.0
    publication_info: float = 0.0
    
    # Engagement factors
    social_proof: float = 0.0
    multimedia_content: float = 0.0
    
    # Technical factors
    dom_complexity: float = 0.0
    performance_impact: float = 0.0
    
    def get_weighted_score(self, weights: Optional[Dict[str, float]] = None) -> float:
        """Calculate weighted quality score."""
        if weights is None:
            weights = {
                'content_length': 0.15,
                'readability': 0.15,
                'lexical_diversity': 0.10,
                'grammar_quality': 0.10,
                'html_structure': 0.10,
                'semantic_markup': 0.08,
                'accessibility': 0.07,
                'metadata_completeness': 0.08,
                'author_credibility': 0.05,
                'publication_info': 0.05,
                'social_proof': 0.03,
                'multimedia_content': 0.02,
                'dom_complexity': 0.01,
                'performance_impact': 0.01,
            }
        
        total_score = 0.0
        total_weight = 0.0
        
        for factor, weight in weights.items():
            if hasattr(self, factor):
                factor_value = getattr(self, factor)
                total_score += factor_value * weight
                total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0


@dataclass
class QualityScore:
    """Comprehensive quality score with detailed breakdown."""
    
    # Overall scores
    overall_score: float = 0.0
    confidence: float = 0.0
    
    # Category scores
    content_quality: float = 0.0
    technical_quality: float = 0.0
    metadata_quality: float = 0.0
    engagement_quality: float = 0.0
    
    # Detailed factors
    factors: QualityFactors = field(default_factory=QualityFactors)
    
    # Quality indicators
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    # Scoring metadata
    scoring_method: str = "comprehensive"
    factors_considered: int = 0
    
    def get_quality_grade(self) -> str:
        """Get letter grade based on overall score."""
        if self.overall_score >= 0.9:
            return 'A+'
        elif self.overall_score >= 0.8:
            return 'A'
        elif self.overall_score >= 0.7:
            return 'B'
        elif self.overall_score >= 0.6:
            return 'C'
        elif self.overall_score >= 0.5:
            return 'D'
        else:
            return 'F'
    
    def get_quality_description(self) -> str:
        """Get descriptive quality assessment."""
        score = self.overall_score
        
        if score >= 0.9:
            return 'Excellent - High-quality content suitable for premium datasets'
        elif score >= 0.8:
            return 'Very Good - Quality content with minor areas for improvement'
        elif score >= 0.7:
            return 'Good - Solid content quality with some optimization opportunities'
        elif score >= 0.6:
            return 'Fair - Acceptable quality but needs improvement'
        elif score >= 0.5:
            return 'Poor - Below average quality, significant improvements needed'
        else:
            return 'Very Poor - Low quality content, major issues present'


class QualityScorer:
    """
    Comprehensive quality scoring system.
    
    Combines metrics from all metadata extraction components
    to provide detailed quality assessment and scoring.
    """
    
    def __init__(self) -> None:
        # Quality thresholds for different factors
        self.thresholds = {
            'excellent': 0.9,
            'very_good': 0.8,
            'good': 0.7,
            'fair': 0.6,
            'poor': 0.5,
        }
        
        # Minimum requirements for different quality levels
        self.requirements = {
            'premium': {
                'min_word_count': 300,
                'min_readability': 0.6,
                'requires_author': True,
                'requires_date': True,
                'min_metadata_completeness': 0.8,
            },
            'standard': {
                'min_word_count': 150,
                'min_readability': 0.5,
                'requires_author': False,
                'requires_date': False,
                'min_metadata_completeness': 0.5,
            },
            'basic': {
                'min_word_count': 50,
                'min_readability': 0.3,
                'requires_author': False,
                'requires_date': False,
                'min_metadata_completeness': 0.3,
            }
        }
        
        logger.info("QualityScorer initialized")
    
    async def calculate_overall_quality(
        self,
        metadata,  # ContentMetadata object
        html_content: str = "",
        text_content: str = "",
        *,
        custom_weights: Optional[Dict[str, float]] = None,
    ) -> QualityScore:
        """
        Calculate comprehensive quality score.
        
        Args:
            metadata: ContentMetadata object with all extracted data
            html_content: Original HTML content
            text_content: Plain text content
            custom_weights: Custom weighting for quality factors
            
        Returns:
            QualityScore with detailed assessment
        """
        quality_score = QualityScore()
        
        try:
            # Calculate individual quality factors
            factors = await self._calculate_quality_factors(metadata, html_content, text_content)
            quality_score.factors = factors
            
            # Calculate category scores
            quality_score.content_quality = self._calculate_content_quality(factors)
            quality_score.technical_quality = self._calculate_technical_quality(factors)
            quality_score.metadata_quality = self._calculate_metadata_quality(factors)
            quality_score.engagement_quality = self._calculate_engagement_quality(factors)
            
            # Calculate overall score
            quality_score.overall_score = factors.get_weighted_score(custom_weights)
            
            # Calculate confidence based on available data
            quality_score.confidence = self._calculate_confidence(metadata, factors)
            
            # Identify strengths and weaknesses
            quality_score.strengths = self._identify_strengths(factors)
            quality_score.weaknesses = self._identify_weaknesses(factors)
            quality_score.recommendations = self._generate_recommendations(factors, metadata)
            
            # Set scoring metadata
            quality_score.scoring_method = "comprehensive"
            quality_score.factors_considered = self._count_available_factors(factors)
            
        except Exception as e:
            logger.error(f"Quality scoring error: {e}")
            # Return minimal score on error
            quality_score.overall_score = 0.1
            quality_score.confidence = 0.1
        
        return quality_score
    
    async def _calculate_quality_factors(self, metadata: Any, html_content: str, text_content: str) -> QualityFactors:
        """Calculate individual quality factors."""
        factors = QualityFactors()
        
        # Content length factor
        word_count = getattr(metadata, 'word_count', 0)
        factors.content_length = self._score_content_length(word_count)
        
        # Readability factor
        reading_metrics = getattr(metadata, 'reading_metrics', None)
        if reading_metrics and hasattr(reading_metrics, 'flesch_reading_ease'):
            factors.readability = self._score_readability(reading_metrics.flesch_reading_ease)
        
        # Lexical diversity factor
        lexical_metrics = getattr(metadata, 'lexical_metrics', None)
        if lexical_metrics and hasattr(lexical_metrics, 'lexical_diversity'):
            factors.lexical_diversity = lexical_metrics.lexical_diversity
        
        # Grammar quality (estimated from various indicators)
        quality_indicators = getattr(metadata, 'quality_indicators', None)
        if quality_indicators:
            factors.grammar_quality = self._estimate_grammar_quality(quality_indicators)
        
        # HTML structure quality
        dom_metrics = getattr(metadata, 'dom_metrics', None)
        if dom_metrics:
            factors.html_structure = self._score_html_structure(dom_metrics)
            factors.semantic_markup = self._score_semantic_markup(dom_metrics)
            factors.accessibility = getattr(dom_metrics, 'accessibility_score', 0.0)
            factors.dom_complexity = getattr(dom_metrics, 'complexity_score', 0.0)
            factors.performance_impact = getattr(dom_metrics, 'dom_size_score', 0.0)
        
        # Metadata completeness
        factors.metadata_completeness = self._score_metadata_completeness(metadata)
        
        # Author credibility
        authors = getattr(metadata, 'authors', [])
        factors.author_credibility = self._score_author_credibility(authors)
        
        # Publication info quality
        factors.publication_info = self._score_publication_info(metadata)
        
        # Social proof
        social_metrics = getattr(metadata, 'social_metrics', None)
        if social_metrics:
            factors.social_proof = self._score_social_proof(social_metrics)
        
        # Multimedia content
        factors.multimedia_content = self._score_multimedia_content(metadata, html_content)
        
        return factors
    
    def _score_content_length(self, word_count: int) -> float:
        """Score content based on length."""
        if word_count >= 1000:
            return 1.0
        elif word_count >= 500:
            return 0.9
        elif word_count >= 300:
            return 0.8
        elif word_count >= 150:
            return 0.6
        elif word_count >= 50:
            return 0.4
        else:
            return 0.2
    
    def _score_readability(self, flesch_score: Optional[float]) -> float:
        """Score content readability."""
        if flesch_score is None:
            return 0.5  # Neutral score when unavailable
        
        # Optimal readability is around 40-70 on Flesch scale
        if 40 <= flesch_score <= 70:
            return 1.0
        elif 30 <= flesch_score <= 80:
            return 0.8
        elif 20 <= flesch_score <= 90:
            return 0.6
        elif 10 <= flesch_score <= 95:
            return 0.4
        else:
            return 0.2
    
    def _estimate_grammar_quality(self, quality_indicators: Any) -> float:
        """Estimate grammar quality from quality indicators."""
        score = 0.0
        factors = 0
        
        if hasattr(quality_indicators, 'proper_capitalization'):
            score += 1.0 if quality_indicators.proper_capitalization else 0.0
            factors += 1
        
        if hasattr(quality_indicators, 'proper_punctuation'):
            score += 1.0 if quality_indicators.proper_punctuation else 0.0
            factors += 1
        
        if hasattr(quality_indicators, 'minimal_typos'):
            score += 1.0 if quality_indicators.minimal_typos else 0.0
            factors += 1
        
        if hasattr(quality_indicators, 'coherent_structure'):
            score += 1.0 if quality_indicators.coherent_structure else 0.0
            factors += 1
        
        return score / factors if factors > 0 else 0.5
    
    def _score_html_structure(self, dom_metrics: Any) -> float:
        """Score HTML structure quality."""
        score = 0.0
        
        # Proper heading hierarchy
        if getattr(dom_metrics, 'has_proper_headings', False):
            score += 0.3
        
        # Semantic HTML usage
        if getattr(dom_metrics, 'has_semantic_html', False):
            score += 0.3
        
        # Good text-to-HTML ratio
        text_ratio = getattr(dom_metrics, 'text_to_html_ratio', 0.0)
        if text_ratio >= 0.2:
            score += 0.2
        elif text_ratio >= 0.1:
            score += 0.1
        
        # Reasonable DOM complexity
        complexity = getattr(dom_metrics, 'complexity_score', 0.0)
        score += complexity * 0.2
        
        return min(1.0, score)
    
    def _score_semantic_markup(self, dom_metrics: Any) -> float:
        """Score semantic markup usage."""
        score = 0.0
        
        if getattr(dom_metrics, 'has_semantic_html', False):
            score += 0.5
        
        structure_metrics = getattr(dom_metrics, 'structure_metrics', None)
        if structure_metrics:
            if getattr(structure_metrics, 'has_semantic_structure', False):
                score += 0.3
            
            semantic_elements = getattr(structure_metrics, 'semantic_elements', [])
            if len(semantic_elements) >= 3:
                score += 0.2
            elif len(semantic_elements) >= 1:
                score += 0.1
        
        return min(1.0, score)
    
    def _score_metadata_completeness(self, metadata: Any) -> float:
        """Score metadata completeness."""
        score = 0.0
        total_factors = 8  # Total metadata factors to check
        
        # Basic metadata
        if getattr(metadata, 'title', None):
            score += 1
        if getattr(metadata, 'description', None):
            score += 1
        if getattr(metadata, 'authors', None):
            score += 1
        if getattr(metadata, 'publication_date', None):
            score += 1
        
        # Structured data
        if getattr(metadata, 'structured_data', None):
            score += 1
        
        # Content categorization
        if getattr(metadata, 'content_categories', None):
            score += 1
        
        # Quality indicators
        quality_indicators = getattr(metadata, 'quality_indicators', None)
        if quality_indicators and getattr(quality_indicators, 'meta_completeness', 0) > 0.5:
            score += 1
        
        # Additional metadata
        if getattr(metadata, 'featured_image', None):
            score += 1
        
        return score / total_factors
    
    def _score_author_credibility(self, authors: List[Any]) -> float:
        """Score author credibility."""
        if not authors:
            return 0.0
        
        # Take the highest confidence author
        best_author = max(authors, key=lambda a: getattr(a, 'confidence_score', 0.0))
        
        confidence = getattr(best_author, 'confidence_score', 0.0)
        
        # Bonus for additional author information
        bonus = 0.0
        if getattr(best_author, 'email', None):
            bonus += 0.1
        if getattr(best_author, 'url', None):
            bonus += 0.1
        if getattr(best_author, 'bio', None):
            bonus += 0.1
        
        return min(1.0, confidence + bonus)
    
    def _score_publication_info(self, metadata: Any) -> float:
        """Score publication information quality."""
        score = 0.0
        
        # Publication date
        if getattr(metadata, 'publication_date', None):
            score += 0.5
            
            # Bonus for date confidence
            date_confidence = getattr(metadata, 'date_confidence', 0.0)
            score += date_confidence * 0.3
        
        # URL quality
        url = getattr(metadata, 'url', '')
        if url and self._is_quality_url(url):
            score += 0.2
        
        return min(1.0, score)
    
    def _score_social_proof(self, social_metrics: Any) -> float:
        """Score social proof indicators."""
        score = 0.0
        
        # Engagement metrics
        total_engagement = getattr(social_metrics, 'total_engagement', 0)
        if total_engagement > 100:
            score += 0.4
        elif total_engagement > 10:
            score += 0.2
        elif total_engagement > 0:
            score += 0.1
        
        # Social sharing capabilities
        if getattr(social_metrics, 'has_social_sharing', False):
            score += 0.2
        
        # Multiple platforms
        platforms = getattr(social_metrics, 'platforms', {})
        if len(platforms) >= 3:
            score += 0.3
        elif len(platforms) >= 1:
            score += 0.1
        
        return min(1.0, score)
    
    def _score_multimedia_content(self, metadata: Any, html_content: str) -> float:
        """Score multimedia content presence."""
        score = 0.0
        
        # Featured image
        if getattr(metadata, 'featured_image', None):
            score += 0.3
        
        # DOM metrics for media
        dom_metrics = getattr(metadata, 'dom_metrics', None)
        if dom_metrics:
            structure_metrics = getattr(dom_metrics, 'structure_metrics', None)
            if structure_metrics:
                if getattr(structure_metrics, 'images', 0) > 0:
                    score += 0.4
                if getattr(structure_metrics, 'videos', 0) > 0:
                    score += 0.2
                if getattr(structure_metrics, 'audio', 0) > 0:
                    score += 0.1
        
        return min(1.0, score)
    
    def _calculate_content_quality(self, factors: QualityFactors) -> float:
        """Calculate content quality category score."""
        content_factors = [
            factors.content_length,
            factors.readability,
            factors.lexical_diversity,
            factors.grammar_quality,
        ]
        
        available_factors = [f for f in content_factors if f > 0]
        return sum(available_factors) / len(available_factors) if available_factors else 0.0
    
    def _calculate_technical_quality(self, factors: QualityFactors) -> float:
        """Calculate technical quality category score."""
        technical_factors = [
            factors.html_structure,
            factors.semantic_markup,
            factors.accessibility,
            factors.dom_complexity,
            factors.performance_impact,
        ]
        
        available_factors = [f for f in technical_factors if f > 0]
        return sum(available_factors) / len(available_factors) if available_factors else 0.0
    
    def _calculate_metadata_quality(self, factors: QualityFactors) -> float:
        """Calculate metadata quality category score."""
        metadata_factors = [
            factors.metadata_completeness,
            factors.author_credibility,
            factors.publication_info,
        ]
        
        available_factors = [f for f in metadata_factors if f > 0]
        return sum(available_factors) / len(available_factors) if available_factors else 0.0
    
    def _calculate_engagement_quality(self, factors: QualityFactors) -> float:
        """Calculate engagement quality category score."""
        engagement_factors = [
            factors.social_proof,
            factors.multimedia_content,
        ]
        
        available_factors = [f for f in engagement_factors if f > 0]
        return sum(available_factors) / len(available_factors) if available_factors else 0.0
    
    def _calculate_confidence(self, metadata: Any, factors: QualityFactors) -> float:
        """Calculate confidence in quality assessment."""
        confidence = 0.0
        
        # Base confidence from available data
        available_factors = self._count_available_factors(factors)
        total_factors = 14  # Total number of quality factors
        data_completeness = available_factors / total_factors
        confidence += data_completeness * 0.5
        
        # Confidence from extraction methods
        if hasattr(metadata, 'structured_data') and metadata.structured_data:
            confidence += 0.2
        
        if hasattr(metadata, 'authors') and metadata.authors:
            confidence += 0.1
        
        if hasattr(metadata, 'publication_date') and metadata.publication_date:
            confidence += 0.1
        
        # Confidence from content analysis
        if hasattr(metadata, 'word_count') and metadata.word_count > 100:
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def _count_available_factors(self, factors: QualityFactors) -> int:
        """Count how many quality factors have data."""
        factor_names = [
            'content_length', 'readability', 'lexical_diversity', 'grammar_quality',
            'html_structure', 'semantic_markup', 'accessibility',
            'metadata_completeness', 'author_credibility', 'publication_info',
            'social_proof', 'multimedia_content', 'dom_complexity', 'performance_impact'
        ]
        
        return sum(1 for name in factor_names if getattr(factors, name, 0) > 0)
    
    def _identify_strengths(self, factors: QualityFactors) -> List[str]:
        """Identify content strengths based on quality factors."""
        strengths = []
        threshold = 0.8  # High quality threshold
        
        factor_descriptions = {
            'content_length': 'Excellent content length',
            'readability': 'Highly readable content',
            'lexical_diversity': 'Rich vocabulary usage',
            'grammar_quality': 'Excellent grammar and writing quality',
            'html_structure': 'Well-structured HTML',
            'semantic_markup': 'Proper semantic markup',
            'accessibility': 'Good accessibility features',
            'metadata_completeness': 'Comprehensive metadata',
            'author_credibility': 'Credible author information',
            'publication_info': 'Complete publication details',
            'social_proof': 'Strong social engagement',
            'multimedia_content': 'Rich multimedia content',
        }
        
        for factor_name, description in factor_descriptions.items():
            factor_value = getattr(factors, factor_name, 0.0)
            if factor_value >= threshold:
                strengths.append(description)
        
        return strengths
    
    def _identify_weaknesses(self, factors: QualityFactors) -> List[str]:
        """Identify content weaknesses based on quality factors."""
        weaknesses = []
        threshold = 0.3  # Low quality threshold
        
        factor_descriptions = {
            'content_length': 'Content too short',
            'readability': 'Poor readability',
            'lexical_diversity': 'Limited vocabulary',
            'grammar_quality': 'Grammar and writing issues',
            'html_structure': 'Poor HTML structure',
            'semantic_markup': 'Lacks semantic markup',
            'accessibility': 'Accessibility issues',
            'metadata_completeness': 'Incomplete metadata',
            'author_credibility': 'Missing or unclear author information',
            'publication_info': 'Missing publication details',
            'social_proof': 'No social engagement',
            'multimedia_content': 'Lacks multimedia content',
        }
        
        for factor_name, description in factor_descriptions.items():
            factor_value = getattr(factors, factor_name, 0.0)
            if factor_value <= threshold:
                weaknesses.append(description)
        
        return weaknesses
    
    def _generate_recommendations(self, factors: QualityFactors, metadata: Any) -> List[str]:
        """Generate improvement recommendations."""
        recommendations = []
        
        # Content recommendations
        if factors.content_length < 0.5:
            recommendations.append("Expand content length for better depth and value")
        
        if factors.readability < 0.6:
            recommendations.append("Improve readability with shorter sentences and simpler language")
        
        if factors.lexical_diversity < 0.4:
            recommendations.append("Use more varied vocabulary to improve lexical diversity")
        
        # Technical recommendations
        if factors.html_structure < 0.6:
            recommendations.append("Improve HTML structure with proper headings and semantic elements")
        
        if factors.accessibility < 0.6:
            recommendations.append("Add alt text for images and improve accessibility features")
        
        # Metadata recommendations
        if factors.metadata_completeness < 0.6:
            recommendations.append("Add missing metadata like author, publication date, and descriptions")
        
        if factors.author_credibility < 0.5:
            recommendations.append("Provide clear author information and credentials")
        
        # Engagement recommendations
        if factors.social_proof < 0.3:
            recommendations.append("Add social sharing buttons and engagement features")
        
        if factors.multimedia_content < 0.3:
            recommendations.append("Include relevant images, videos, or other multimedia content")
        
        return recommendations[:5]  # Return top 5 recommendations
    
    def _is_quality_url(self, url: str) -> bool:
        """Check if URL indicates quality content."""
        quality_indicators = [
            '.edu', '.gov', '.org',  # Institutional domains
            'blog', 'article', 'post',  # Content-focused paths
        ]
        
        return any(indicator in url.lower() for indicator in quality_indicators)
    
    async def assess_content_tier(self, quality_score: QualityScore) -> str:
        """Assess content tier based on quality score and requirements."""
        score = quality_score.overall_score
        
        # Check premium tier requirements
        if score >= 0.8 and self._meets_premium_requirements(quality_score):
            return 'premium'
        
        # Check standard tier requirements
        elif score >= 0.6 and self._meets_standard_requirements(quality_score):
            return 'standard'
        
        # Check basic tier requirements
        elif score >= 0.4 and self._meets_basic_requirements(quality_score):
            return 'basic'
        
        else:
            return 'below_threshold'
    
    def _meets_premium_requirements(self, quality_score: QualityScore) -> bool:
        """Check if content meets premium tier requirements."""
        requirements = self.requirements['premium']
        
        # Check individual requirements
        checks = [
            quality_score.content_quality >= 0.8,
            quality_score.metadata_quality >= 0.7,
            quality_score.confidence >= 0.8,
        ]
        
        return all(checks)
    
    def _meets_standard_requirements(self, quality_score: QualityScore) -> bool:
        """Check if content meets standard tier requirements."""
        requirements = self.requirements['standard']
        
        checks = [
            quality_score.content_quality >= 0.6,
            quality_score.confidence >= 0.6,
        ]
        
        return all(checks)
    
    def _meets_basic_requirements(self, quality_score: QualityScore) -> bool:
        """Check if content meets basic tier requirements."""
        requirements = self.requirements['basic']
        
        checks = [
            quality_score.overall_score >= 0.4,
            quality_score.confidence >= 0.4,
        ]
        
        return all(checks) 