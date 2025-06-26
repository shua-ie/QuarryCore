"""
DOM Analyzer - HTML Structure and Complexity Analysis

Analyzes DOM structure including depth, node count, text ratio, and complexity
metrics for comprehensive content structure evaluation.
"""

from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

# BeautifulSoup imports with graceful fallbacks (proven pattern)
try:
    from bs4 import BeautifulSoup, NavigableString, Tag
    HAS_BS4 = True
except ImportError:
    if TYPE_CHECKING:
        from bs4 import BeautifulSoup, NavigableString, Tag
    else:
        BeautifulSoup = None  # type: ignore[misc,assignment]
        NavigableString = None  # type: ignore[misc,assignment]
        Tag = None  # type: ignore[misc,assignment]
    HAS_BS4 = False

logger = logging.getLogger(__name__)


@dataclass
class DOMAnalysisConfig:
    """Configuration for DOM analysis."""
    
    # Analysis depth limits
    max_analysis_depth: int = 50
    max_nodes_to_analyze: int = 10000
    
    # Content analysis settings
    analyze_accessibility: bool = True
    analyze_performance: bool = True
    analyze_seo: bool = True
    
    # Quality thresholds
    min_text_ratio: float = 0.1
    max_dom_depth: int = 15
    max_elements: int = 1500
    
    # Timeout settings
    analysis_timeout_seconds: float = 30.0


@dataclass
class StructureMetrics:
    """HTML structure and organization metrics."""
    
    # Hierarchy metrics
    max_depth: int = 0
    avg_depth: float = 0.0
    depth_distribution: Dict[int, int] = field(default_factory=dict)
    
    # Content organization
    has_semantic_structure: bool = False
    semantic_elements: List[str] = field(default_factory=list)
    heading_hierarchy: List[str] = field(default_factory=list)
    
    # Navigation and linking
    internal_links: int = 0
    external_links: int = 0
    broken_links: int = 0
    
    # Media content
    images: int = 0
    videos: int = 0
    audio: int = 0
    
    # Interactive elements
    forms: int = 0
    buttons: int = 0
    inputs: int = 0


@dataclass
class DOMMetrics:
    """Comprehensive DOM analysis metrics."""
    
    # Basic structure metrics
    total_nodes: int = 0
    element_nodes: int = 0
    text_nodes: int = 0
    comment_nodes: int = 0
    
    # Content metrics
    total_text_length: int = 0
    visible_text_length: int = 0
    text_to_html_ratio: float = 0.0
    
    # Complexity metrics
    max_depth: int = 0
    avg_depth: float = 0.0
    branching_factor: float = 0.0
    
    # Element distribution
    element_counts: Dict[str, int] = field(default_factory=dict)
    class_usage: Dict[str, int] = field(default_factory=dict)
    id_usage: Set[str] = field(default_factory=set)
    
    # Structure quality
    structure_metrics: StructureMetrics = field(default_factory=StructureMetrics)
    
    # Performance indicators
    dom_size_score: float = 0.0      # 0-1, higher is better
    complexity_score: float = 0.0    # 0-1, lower complexity is better
    accessibility_score: float = 0.0  # 0-1, higher is better
    
    # Quality indicators
    has_proper_headings: bool = False
    has_alt_text: bool = False
    has_semantic_html: bool = False
    has_proper_forms: bool = False


class DOMAnalyzer:
    """
    DOM structure and complexity analyzer.
    
    Analyzes HTML structure for complexity, organization,
    accessibility, and performance characteristics.
    """
    
    def __init__(self, config: Optional[DOMAnalysisConfig] = None) -> None:
        self.config = config or DOMAnalysisConfig()
        
        # Initialize BeautifulSoup parser if available
        self.has_bs4 = HAS_BS4
        if not self.has_bs4:
            logger.warning("BeautifulSoup not available, DOM analysis limited")
        
        # Cache for parsed documents
        self.parsed_cache: Dict[str, Any] = {}
        
        # Semantic HTML elements
        self.semantic_elements: Set[str] = {
            'article', 'aside', 'details', 'figcaption', 'figure',
            'footer', 'header', 'main', 'mark', 'nav', 'section',
            'summary', 'time'
        }
        
        # Heading elements
        self.heading_elements: Set[str] = {'h1', 'h2', 'h3', 'h4', 'h5', 'h6'}
        
        # Interactive elements
        self.interactive_elements: Set[str] = {
            'a', 'button', 'input', 'select', 'textarea', 'label',
            'form', 'fieldset', 'legend'
        }
        
        # Media elements
        self.media_elements: Set[str] = {
            'img', 'video', 'audio', 'canvas', 'svg', 'picture',
            'source', 'track'
        }
        
        # Elements that typically don't contribute to visible content
        self.non_content_elements: Set[str] = {
            'script', 'style', 'noscript', 'template', 'meta',
            'link', 'title', 'head'
        }
        
        logger.info("DOMAnalyzer initialized")
    
    async def analyze_structure(self, html_content: str) -> DOMMetrics:
        """
        Analyze DOM structure and complexity.
        
        Args:
            html_content: HTML content to analyze
            
        Returns:
            DOMMetrics with comprehensive structure analysis
        """
        metrics = DOMMetrics()
        
        if not html_content:
            return metrics
        
        try:
            if self.has_bs4:
                await self._analyze_with_beautifulsoup(html_content, metrics)
            else:
                await self._analyze_with_regex(html_content, metrics)
            
            # Calculate derived metrics
            self._calculate_derived_metrics(metrics)
            
            # Calculate quality scores
            self._calculate_quality_scores(metrics)
        
        except Exception as e:
            logger.error(f"DOM analysis error: {e}")
        
        return metrics
    
    async def _analyze_with_beautifulsoup(self, html_content: str, metrics: DOMMetrics) -> None:
        """Analyze DOM using BeautifulSoup."""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Basic node counting
        self._count_nodes(soup, metrics)
        
        # Analyze structure depth and complexity
        self._analyze_depth_complexity(soup, metrics)
        
        # Analyze element distribution
        self._analyze_element_distribution(soup, metrics)
        
        # Analyze content and text
        self._analyze_content_metrics(soup, metrics, html_content)
        
        # Analyze structure quality
        await self._analyze_structure_quality(soup, metrics)
    
    async def _analyze_with_regex(self, html_content: str, metrics: DOMMetrics) -> None:
        """Fallback DOM analysis using regex."""
        # Count elements using regex
        element_pattern = r'<(\w+)(?:\s[^>]*)?>'
        element_matches = re.finditer(element_pattern, html_content, re.IGNORECASE)
        
        element_counts: Dict[str, int] = {}
        for match in element_matches:
            tag_name = match.group(1).lower()
            element_counts[tag_name] = element_counts.get(tag_name, 0) + 1
        
        metrics.element_counts = element_counts
        metrics.element_nodes = sum(element_counts.values())
        
        # Estimate text content
        text_content = re.sub(r'<[^>]+>', ' ', html_content)
        text_content = re.sub(r'\s+', ' ', text_content).strip()
        metrics.visible_text_length = len(text_content)
        metrics.total_text_length = metrics.visible_text_length
        
        # Calculate basic ratios
        html_size = len(html_content)
        if html_size > 0:
            metrics.text_to_html_ratio = metrics.visible_text_length / html_size
        
        # Basic structure analysis
        self._analyze_structure_regex(html_content, metrics)
    
    def _count_nodes(self, soup: Any, metrics: DOMMetrics) -> None:
        """Count different types of nodes in the DOM."""
        def count_recursive(element: Any, depth: int = 0) -> None:
            if HAS_BS4 and hasattr(element, 'name') and element.name:  # Tag element
                metrics.element_nodes += 1
                metrics.total_nodes += 1
                
                # Track element types
                tag_name = str(element.name).lower()
                element_counts = metrics.element_counts
                element_counts[tag_name] = element_counts.get(tag_name, 0) + 1
                
                # Track classes and IDs
                if hasattr(element, 'get'):
                    class_list = element.get('class')
                    if class_list:
                        if isinstance(class_list, list):
                            for class_name in class_list:
                                class_usage = metrics.class_usage
                                class_usage[str(class_name)] = class_usage.get(str(class_name), 0) + 1
                        else:
                            class_usage = metrics.class_usage
                            class_usage[str(class_list)] = class_usage.get(str(class_list), 0) + 1
                    
                    element_id = element.get('id')
                    if element_id:
                        metrics.id_usage.add(str(element_id))
                
                # Recurse through children
                if hasattr(element, 'children'):
                    for child in element.children:
                        count_recursive(child, depth + 1)
            
            elif isinstance(element, str) or (HAS_BS4 and hasattr(element, 'string')):
                # Text node
                text_content = str(element).strip() if isinstance(element, str) else str(element.string or '').strip()
                if text_content:  # Only count non-empty text nodes
                    metrics.text_nodes += 1
                    metrics.total_nodes += 1
        
        count_recursive(soup)
    
    def _analyze_depth_complexity(self, soup: Any, metrics: DOMMetrics) -> None:
        """Analyze DOM depth and structural complexity."""
        depths: List[int] = []
        child_counts: List[int] = []
        
        def analyze_recursive(element: Any, depth: int = 0) -> None:
            if HAS_BS4 and hasattr(element, 'name') and element.name:  # Tag element
                depths.append(depth)
                
                # Count direct children
                direct_children = 0
                if hasattr(element, 'children'):
                    for child in element.children:
                        if hasattr(child, 'name') and child.name:
                            direct_children += 1
                child_counts.append(direct_children)
                
                # Update structure metrics
                depth_dist = metrics.structure_metrics.depth_distribution
                depth_dist[depth] = depth_dist.get(depth, 0) + 1
                
                # Recurse
                if hasattr(element, 'children'):
                    for child in element.children:
                        analyze_recursive(child, depth + 1)
        
        analyze_recursive(soup)
        
        if depths:
            metrics.max_depth = max(depths)
            metrics.avg_depth = sum(depths) / len(depths)
            metrics.structure_metrics.max_depth = metrics.max_depth
            metrics.structure_metrics.avg_depth = metrics.avg_depth
        
        if child_counts:
            metrics.branching_factor = sum(child_counts) / len(child_counts)
    
    def _analyze_element_distribution(self, soup: Any, metrics: DOMMetrics) -> None:
        """Analyze distribution and usage of HTML elements."""
        if not HAS_BS4:
            return
            
        # Count semantic elements
        semantic_count = 0
        for element in self.semantic_elements:
            found_elements = soup.find_all(element)
            count = len(found_elements) if found_elements else 0
            if count > 0:
                semantic_count += count
                metrics.structure_metrics.semantic_elements.append(element)
        
        metrics.structure_metrics.has_semantic_structure = semantic_count > 0
        
        # Analyze heading hierarchy
        headings = soup.find_all(list(self.heading_elements))
        heading_levels: List[int] = []
        for heading in headings:
            if hasattr(heading, 'name') and heading.name:
                level = int(heading.name[1])  # Extract number from h1, h2, etc.
                heading_levels.append(level)
                metrics.structure_metrics.heading_hierarchy.append(str(heading.name))
        
        # Check for proper heading hierarchy
        if heading_levels:
            metrics.has_proper_headings = self._check_heading_hierarchy(heading_levels)
        
        # Count links
        links = soup.find_all('a')
        for link in links:
            if hasattr(link, 'get'):
                href = link.get('href')
                if href:
                    href_str = str(href).lower()
                    if href_str.startswith('http') or href_str.startswith('//'):
                        metrics.structure_metrics.external_links += 1
                    elif href_str.startswith('#') or href_str.startswith('/') or not href_str.startswith('http'):
                        metrics.structure_metrics.internal_links += 1
        
        # Count media elements
        metrics.structure_metrics.images = len(soup.find_all('img') or [])
        metrics.structure_metrics.videos = len(soup.find_all('video') or [])
        metrics.structure_metrics.audio = len(soup.find_all('audio') or [])
        
        # Count interactive elements
        metrics.structure_metrics.forms = len(soup.find_all('form') or [])
        metrics.structure_metrics.buttons = len(soup.find_all('button') or [])
        metrics.structure_metrics.inputs = len(soup.find_all('input') or [])
    
    def _analyze_content_metrics(self, soup: Any, metrics: DOMMetrics, html_content: str) -> None:
        """Analyze content and text metrics."""
        if not HAS_BS4:
            return
            
        # Get all text content
        all_text = soup.get_text() if hasattr(soup, 'get_text') else ''
        metrics.total_text_length = len(str(all_text))
        
        # Get visible text (excluding script, style, etc.)
        for element_name in self.non_content_elements:
            elements_to_remove = soup.find_all(element_name)
            for element in elements_to_remove:
                if hasattr(element, 'decompose'):
                    element.decompose()
        
        visible_text = soup.get_text() if hasattr(soup, 'get_text') else ''
        metrics.visible_text_length = len(str(visible_text))
        
        # Calculate text-to-HTML ratio
        html_size = len(html_content)
        if html_size > 0:
            metrics.text_to_html_ratio = metrics.visible_text_length / html_size
    
    async def _analyze_structure_quality(self, soup: Any, metrics: DOMMetrics) -> None:
        """Analyze structure quality and accessibility."""
        if not HAS_BS4:
            return
            
        # Check for semantic HTML usage
        semantic_elements_found = False
        for element in self.semantic_elements:
            if soup.find_all(element):
                semantic_elements_found = True
                break
        metrics.has_semantic_html = semantic_elements_found
        
        # Check for alt text on images
        images = soup.find_all('img') or []
        images_with_alt = 0
        for img in images:
            if hasattr(img, 'get') and img.get('alt'):
                images_with_alt += 1
        metrics.has_alt_text = len(images) == 0 or images_with_alt / len(images) > 0.8
        
        # Check for proper form structure
        forms = soup.find_all('form') or []
        proper_forms = 0
        for form in forms:
            # Check if form has labels for inputs
            inputs = form.find_all(['input', 'select', 'textarea']) or []
            labels = form.find_all('label') or []
            if len(labels) >= len(inputs) * 0.8:  # 80% of inputs have labels
                proper_forms += 1
        
        metrics.has_proper_forms = len(forms) == 0 or proper_forms / len(forms) > 0.8
    
    def _analyze_structure_regex(self, html_content: str, metrics: DOMMetrics) -> None:
        """Basic structure analysis using regex."""
        # Check for semantic elements
        semantic_found = any(
            re.search(f'<{element}[^>]*>', html_content, re.IGNORECASE)
            for element in self.semantic_elements
        )
        metrics.has_semantic_html = semantic_found
        
        # Check for headings
        heading_pattern = r'<h[1-6][^>]*>'
        headings = re.findall(heading_pattern, html_content, re.IGNORECASE)
        metrics.has_proper_headings = len(headings) > 0
        
        # Check for alt attributes on images
        img_pattern = r'<img[^>]*>'
        alt_pattern = r'<img[^>]*alt=["\'][^"\']*["\'][^>]*>'
        
        images = re.findall(img_pattern, html_content, re.IGNORECASE)
        images_with_alt = re.findall(alt_pattern, html_content, re.IGNORECASE)
        
        metrics.has_alt_text = len(images) == 0 or len(images_with_alt) / len(images) > 0.8
    
    def _check_heading_hierarchy(self, heading_levels: List[int]) -> bool:
        """Check if heading hierarchy is properly structured."""
        if not heading_levels:
            return False
        
        # Should start with h1
        if heading_levels[0] != 1:
            return False
        
        # Check for proper progression (no skipping levels)
        for i in range(1, len(heading_levels)):
            current = heading_levels[i]
            previous = heading_levels[i-1]
            
            # Can only increase by 1 or stay same/decrease
            if current > previous + 1:
                return False
        
        return True
    
    def _calculate_derived_metrics(self, metrics: DOMMetrics) -> None:
        """Calculate derived metrics from basic measurements."""
        # Update structure metrics from main metrics
        metrics.structure_metrics.max_depth = metrics.max_depth
        metrics.structure_metrics.avg_depth = metrics.avg_depth
    
    def _calculate_quality_scores(self, metrics: DOMMetrics) -> None:
        """Calculate quality scores based on DOM analysis."""
        # DOM size score (penalize overly complex DOMs)
        if metrics.element_nodes <= 100:
            metrics.dom_size_score = 1.0
        elif metrics.element_nodes <= 500:
            metrics.dom_size_score = 0.8
        elif metrics.element_nodes <= 1000:
            metrics.dom_size_score = 0.6
        elif metrics.element_nodes <= 2000:
            metrics.dom_size_score = 0.4
        else:
            metrics.dom_size_score = 0.2
        
        # Complexity score (lower is better, so invert for scoring)
        complexity_factors: List[float] = []
        
        # Depth complexity
        if metrics.max_depth <= 10:
            complexity_factors.append(1.0)
        elif metrics.max_depth <= 15:
            complexity_factors.append(0.8)
        elif metrics.max_depth <= 20:
            complexity_factors.append(0.6)
        else:
            complexity_factors.append(0.4)
        
        # Branching factor
        if metrics.branching_factor <= 3:
            complexity_factors.append(1.0)
        elif metrics.branching_factor <= 5:
            complexity_factors.append(0.8)
        elif metrics.branching_factor <= 8:
            complexity_factors.append(0.6)
        else:
            complexity_factors.append(0.4)
        
        # Text-to-HTML ratio (higher is better)
        if metrics.text_to_html_ratio >= 0.3:
            complexity_factors.append(1.0)
        elif metrics.text_to_html_ratio >= 0.2:
            complexity_factors.append(0.8)
        elif metrics.text_to_html_ratio >= 0.1:
            complexity_factors.append(0.6)
        else:
            complexity_factors.append(0.4)
        
        metrics.complexity_score = sum(complexity_factors) / len(complexity_factors)
        
        # Accessibility score
        accessibility_factors: List[float] = []
        
        # Semantic HTML usage
        accessibility_factors.append(1.0 if metrics.has_semantic_html else 0.0)
        
        # Proper headings
        accessibility_factors.append(1.0 if metrics.has_proper_headings else 0.0)
        
        # Alt text on images
        accessibility_factors.append(1.0 if metrics.has_alt_text else 0.0)
        
        # Proper forms
        accessibility_factors.append(1.0 if metrics.has_proper_forms else 0.0)
        
        metrics.accessibility_score = sum(accessibility_factors) / len(accessibility_factors)
    
    async def analyze_performance_impact(self, html_content: str) -> Dict[str, Any]:
        """
        Analyze DOM characteristics that impact performance.
        
        Returns performance analysis with recommendations.
        """
        metrics = await self.analyze_structure(html_content)
        
        analysis: Dict[str, Any] = {
            'dom_size': {
                'element_count': metrics.element_nodes,
                'impact': self._assess_dom_size_impact(metrics.element_nodes),
                'recommendation': self._get_dom_size_recommendation(metrics.element_nodes),
            },
            'dom_depth': {
                'max_depth': metrics.max_depth,
                'avg_depth': metrics.avg_depth,
                'impact': self._assess_depth_impact(metrics.max_depth),
                'recommendation': self._get_depth_recommendation(metrics.max_depth),
            },
            'complexity': {
                'score': metrics.complexity_score,
                'branching_factor': metrics.branching_factor,
                'impact': self._assess_complexity_impact(metrics.complexity_score),
                'recommendation': self._get_complexity_recommendation(metrics.complexity_score),
            },
            'content_ratio': {
                'ratio': metrics.text_to_html_ratio,
                'visible_text_length': metrics.visible_text_length,
                'impact': self._assess_content_ratio_impact(metrics.text_to_html_ratio),
                'recommendation': self._get_content_ratio_recommendation(metrics.text_to_html_ratio),
            },
            'overall_score': {
                'dom_size_score': metrics.dom_size_score,
                'complexity_score': metrics.complexity_score,
                'accessibility_score': metrics.accessibility_score,
            }
        }
        
        return analysis
    
    def _assess_dom_size_impact(self, element_count: int) -> str:
        """Assess performance impact of DOM size."""
        if element_count <= 100:
            return "minimal"
        elif element_count <= 500:
            return "low"
        elif element_count <= 1000:
            return "moderate"
        elif element_count <= 2000:
            return "high"
        else:
            return "severe"
    
    def _get_dom_size_recommendation(self, element_count: int) -> str:
        """Get recommendation for DOM size optimization."""
        if element_count <= 500:
            return "DOM size is optimal"
        elif element_count <= 1000:
            return "Consider reducing DOM complexity"
        else:
            return "Significant DOM optimization needed - consider lazy loading, virtualization, or content splitting"
    
    def _assess_depth_impact(self, max_depth: int) -> str:
        """Assess performance impact of DOM depth."""
        if max_depth <= 10:
            return "minimal"
        elif max_depth <= 15:
            return "low"
        elif max_depth <= 20:
            return "moderate"
        else:
            return "high"
    
    def _get_depth_recommendation(self, max_depth: int) -> str:
        """Get recommendation for DOM depth optimization."""
        if max_depth <= 15:
            return "DOM depth is acceptable"
        else:
            return "Consider flattening DOM structure to improve rendering performance"
    
    def _assess_complexity_impact(self, complexity_score: float) -> str:
        """Assess performance impact of DOM complexity."""
        if complexity_score >= 0.8:
            return "minimal"
        elif complexity_score >= 0.6:
            return "low"
        elif complexity_score >= 0.4:
            return "moderate"
        else:
            return "high"
    
    def _get_complexity_recommendation(self, complexity_score: float) -> str:
        """Get recommendation for complexity optimization."""
        if complexity_score >= 0.7:
            return "DOM complexity is well managed"
        elif complexity_score >= 0.5:
            return "Consider simplifying DOM structure and reducing nesting"
        else:
            return "Significant simplification needed - review component architecture"
    
    def _assess_content_ratio_impact(self, ratio: float) -> str:
        """Assess performance impact of content-to-markup ratio."""
        if ratio >= 0.3:
            return "optimal"
        elif ratio >= 0.2:
            return "good"
        elif ratio >= 0.1:
            return "fair"
        else:
            return "poor"
    
    def _get_content_ratio_recommendation(self, ratio: float) -> str:
        """Get recommendation for content ratio optimization."""
        if ratio >= 0.25:
            return "Excellent content-to-markup ratio"
        elif ratio >= 0.15:
            return "Good content ratio, minor optimization possible"
        else:
            return "Low content ratio - consider reducing markup overhead or increasing meaningful content" 