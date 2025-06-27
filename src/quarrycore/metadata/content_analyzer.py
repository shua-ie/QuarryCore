"""
Content Analyzer - Quality Assessment and Linguistic Analysis

Analyzes content quality, reading time, lexical diversity, and categorization
for comprehensive content evaluation in AI training data mining.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

try:
    from bs4 import BeautifulSoup

    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False
    BeautifulSoup = None  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class ReadingMetrics:
    """Reading-related metrics for content."""

    word_count: int = 0
    character_count: int = 0
    sentence_count: int = 0
    paragraph_count: int = 0

    # Reading time estimates (in minutes)
    reading_time_minutes: float = 0.0
    reading_time_slow: float = 0.0  # 150 WPM
    reading_time_average: float = 0.0  # 200 WPM
    reading_time_fast: float = 0.0  # 250 WPM

    # Readability metrics
    avg_words_per_sentence: float = 0.0
    avg_characters_per_word: float = 0.0
    flesch_reading_ease: Optional[float] = None
    flesch_kincaid_grade: Optional[float] = None


@dataclass
class LexicalMetrics:
    """Lexical diversity and vocabulary metrics."""

    # Basic diversity metrics
    unique_words: int = 0
    total_words: int = 0
    type_token_ratio: float = 0.0  # TTR

    # Advanced diversity metrics
    moving_average_ttr: float = 0.0  # MATTR
    lexical_diversity: float = 0.0  # Combined score

    # Vocabulary complexity
    complex_words: int = 0  # Words with 3+ syllables
    complex_word_ratio: float = 0.0

    # Word frequency analysis
    most_common_words: List[Tuple[str, int]] = field(default_factory=list)
    vocabulary_richness: float = 0.0


@dataclass
class QualityIndicators:
    """Content quality indicators."""

    # Structure indicators
    has_title: bool = False
    has_headings: bool = False
    has_paragraphs: bool = False
    has_lists: bool = False
    has_links: bool = False
    has_images: bool = False

    # Content quality indicators
    has_author: bool = False
    has_date: bool = False
    has_meta_description: bool = False
    has_structured_data: bool = False

    # Text quality indicators
    proper_capitalization: bool = False
    proper_punctuation: bool = False
    minimal_typos: bool = False
    coherent_structure: bool = False

    # Completeness indicators
    meta_completeness: float = 0.0  # 0-1 score
    content_completeness: float = 0.0  # 0-1 score

    # Additional quality factors
    grammar_score: Optional[float] = None
    spelling_score: Optional[float] = None
    coherence_score: Optional[float] = None


@dataclass
class ContentMetrics:
    """Comprehensive content analysis metrics."""

    # Basic metrics
    word_count: int = 0
    reading_time_minutes: float = 0.0
    lexical_diversity: float = 0.0

    # Detailed metrics
    reading_metrics: ReadingMetrics = field(default_factory=ReadingMetrics)
    lexical_metrics: LexicalMetrics = field(default_factory=LexicalMetrics)
    quality_indicators: QualityIndicators = field(default_factory=QualityIndicators)

    # Content categorization
    categories: List[str] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)

    # Overall scores
    quality_score: float = 0.0
    readability_score: float = 0.0
    engagement_score: float = 0.0


class ContentAnalyzer:
    """
    Comprehensive content analysis system.

    Analyzes content quality, readability, lexical diversity,
    and provides categorization for content evaluation.
    """

    def __init__(self) -> None:
        # Stop words for lexical analysis
        self.stop_words = {
            "a",
            "an",
            "and",
            "are",
            "as",
            "at",
            "be",
            "by",
            "for",
            "from",
            "has",
            "he",
            "in",
            "is",
            "it",
            "its",
            "of",
            "on",
            "that",
            "the",
            "to",
            "was",
            "were",
            "will",
            "with",
            "this",
            "but",
            "they",
            "have",
            "had",
            "what",
            "said",
            "each",
            "which",
            "she",
            "do",
            "how",
            "their",
            "if",
            "up",
            "out",
            "many",
            "then",
            "them",
            "these",
            "so",
            "some",
            "her",
            "would",
            "make",
            "like",
            "into",
            "him",
            "time",
            "two",
            "more",
            "go",
            "no",
            "way",
            "could",
            "my",
            "than",
            "first",
            "been",
            "call",
            "who",
            "oil",
            "sit",
            "now",
            "find",
            "down",
            "day",
            "did",
            "get",
            "come",
            "made",
            "may",
            "part",
        }

        # Content category keywords
        self.category_keywords = {
            "technology": [
                "software",
                "hardware",
                "computer",
                "programming",
                "code",
                "api",
                "database",
                "algorithm",
                "machine learning",
                "ai",
                "artificial intelligence",
                "cloud",
                "server",
                "network",
                "security",
                "cyber",
                "digital",
                "mobile",
                "app",
                "application",
                "web",
                "internet",
                "data",
            ],
            "science": [
                "research",
                "study",
                "experiment",
                "analysis",
                "hypothesis",
                "theory",
                "discovery",
                "scientific",
                "biology",
                "chemistry",
                "physics",
                "mathematics",
                "statistics",
                "laboratory",
                "method",
            ],
            "business": [
                "market",
                "company",
                "business",
                "industry",
                "economy",
                "finance",
                "investment",
                "profit",
                "revenue",
                "sales",
                "marketing",
                "strategy",
                "management",
                "corporate",
                "enterprise",
                "startup",
                "entrepreneur",
            ],
            "health": [
                "health",
                "medical",
                "medicine",
                "doctor",
                "patient",
                "treatment",
                "disease",
                "symptom",
                "therapy",
                "hospital",
                "clinical",
                "diagnosis",
                "pharmaceutical",
                "wellness",
                "fitness",
                "nutrition",
            ],
            "education": [
                "education",
                "school",
                "university",
                "student",
                "teacher",
                "learning",
                "curriculum",
                "academic",
                "course",
                "degree",
                "training",
                "knowledge",
                "skill",
                "instruction",
                "pedagogy",
            ],
            "news": [
                "news",
                "report",
                "journalist",
                "media",
                "press",
                "breaking",
                "update",
                "story",
                "coverage",
                "headline",
                "article",
                "newspaper",
                "broadcast",
                "correspondent",
            ],
            "entertainment": [
                "entertainment",
                "movie",
                "film",
                "music",
                "game",
                "sport",
                "celebrity",
                "show",
                "television",
                "streaming",
                "concert",
                "performance",
                "artist",
                "actor",
                "musician",
            ],
            "lifestyle": [
                "lifestyle",
                "fashion",
                "travel",
                "food",
                "recipe",
                "cooking",
                "home",
                "garden",
                "family",
                "relationship",
                "personal",
                "hobby",
                "leisure",
                "culture",
                "art",
            ],
        }

        # Compile category patterns
        self.category_patterns = {}
        for category, keywords in self.category_keywords.items():
            pattern = r"\b(?:" + "|".join(re.escape(kw) for kw in keywords) + r")\b"
            self.category_patterns[category] = re.compile(pattern, re.IGNORECASE)

        logger.info("ContentAnalyzer initialized with comprehensive analysis")

    async def analyze_content(
        self,
        text_content: str,
        html_content: str = "",
        *,
        calculate_reading_metrics: bool = True,
        analyze_lexical_diversity: bool = True,
        categorize_content: bool = True,
    ) -> ContentMetrics:
        """
        Perform comprehensive content analysis.

        Args:
            text_content: Plain text content to analyze
            html_content: HTML content for structure analysis
            calculate_reading_metrics: Whether to calculate reading metrics
            analyze_lexical_diversity: Whether to analyze lexical diversity
            categorize_content: Whether to categorize content

        Returns:
            ContentMetrics with comprehensive analysis results
        """
        metrics = ContentMetrics()

        if not text_content:
            return metrics

        try:
            # Basic word count
            words = self._tokenize_words(text_content)
            metrics.word_count = len(words)

            # Reading metrics
            if calculate_reading_metrics:
                metrics.reading_metrics = await self._calculate_reading_metrics(text_content, words)
                metrics.reading_time_minutes = metrics.reading_metrics.reading_time_average

            # Lexical diversity
            if analyze_lexical_diversity:
                metrics.lexical_metrics = await self._analyze_lexical_diversity(text_content, words)
                metrics.lexical_diversity = metrics.lexical_metrics.lexical_diversity

            # Quality indicators
            metrics.quality_indicators = await self._analyze_quality_indicators(text_content, html_content)

            # Content categorization
            if categorize_content:
                metrics.categories = await self._categorize_content(text_content)
                metrics.topics = await self._extract_topics(text_content)

            # Calculate overall scores
            metrics.quality_score = self._calculate_quality_score(metrics)
            metrics.readability_score = self._calculate_readability_score(metrics)
            metrics.engagement_score = self._calculate_engagement_score(metrics)

        except Exception as e:
            logger.error(f"Content analysis error: {e}")

        return metrics

    def _tokenize_words(self, text: str) -> List[str]:
        """Tokenize text into words."""
        # Simple word tokenization
        words = re.findall(r"\b\w+\b", text.lower())
        return [word for word in words if len(word) > 1]

    async def _calculate_reading_metrics(self, text: str, words: List[str]) -> ReadingMetrics:
        """Calculate reading-related metrics."""
        metrics = ReadingMetrics()

        # Basic counts
        metrics.word_count = len(words)
        metrics.character_count = len(text)

        # Sentence count
        sentences = re.split(r"[.!?]+", text)
        metrics.sentence_count = len([s for s in sentences if s.strip()])

        # Paragraph count
        paragraphs = text.split("\n\n")
        metrics.paragraph_count = len([p for p in paragraphs if p.strip()])

        # Reading time calculations (words per minute)
        if metrics.word_count > 0:
            metrics.reading_time_slow = metrics.word_count / 150  # 150 WPM
            metrics.reading_time_average = metrics.word_count / 200  # 200 WPM
            metrics.reading_time_fast = metrics.word_count / 250  # 250 WPM
            metrics.reading_time_minutes = metrics.reading_time_average

        # Average metrics
        if metrics.sentence_count > 0:
            metrics.avg_words_per_sentence = metrics.word_count / metrics.sentence_count

        if metrics.word_count > 0:
            total_chars = sum(len(word) for word in words)
            metrics.avg_characters_per_word = total_chars / metrics.word_count

        # Readability scores
        metrics.flesch_reading_ease = self._calculate_flesch_reading_ease(
            metrics.word_count, metrics.sentence_count, self._count_syllables(text)
        )

        if metrics.flesch_reading_ease is not None:
            metrics.flesch_kincaid_grade = self._calculate_flesch_kincaid_grade(
                metrics.word_count, metrics.sentence_count, self._count_syllables(text)
            )

        return metrics

    async def _analyze_lexical_diversity(self, text: str, words: List[str]) -> LexicalMetrics:
        """Analyze lexical diversity and vocabulary richness."""
        metrics = LexicalMetrics()

        if not words:
            return metrics

        # Filter out stop words for diversity analysis
        content_words = [word for word in words if word not in self.stop_words]

        # Basic diversity metrics
        metrics.total_words = len(content_words)
        metrics.unique_words = len(set(content_words))

        if metrics.total_words > 0:
            metrics.type_token_ratio = metrics.unique_words / metrics.total_words

        # Moving Average Type-Token Ratio (MATTR)
        if len(content_words) >= 100:
            metrics.moving_average_ttr = self._calculate_mattr(content_words)
        else:
            metrics.moving_average_ttr = metrics.type_token_ratio

        # Combined lexical diversity score
        metrics.lexical_diversity = (metrics.type_token_ratio + metrics.moving_average_ttr) / 2

        # Complex words analysis
        metrics.complex_words = self._count_complex_words(words)
        if metrics.total_words > 0:
            metrics.complex_word_ratio = metrics.complex_words / metrics.total_words

        # Word frequency analysis
        word_freq = Counter(content_words)
        metrics.most_common_words = word_freq.most_common(10)

        # Vocabulary richness (inverse of frequency concentration)
        if word_freq:
            total_freq = sum(word_freq.values())
            freq_concentration = sum((freq / total_freq) ** 2 for freq in word_freq.values())
            metrics.vocabulary_richness = 1 - freq_concentration

        return metrics

    async def _analyze_quality_indicators(self, text: str, html_content: str) -> QualityIndicators:
        """Analyze content quality indicators."""
        indicators = QualityIndicators()

        # Text-based indicators
        indicators.proper_capitalization = self._check_capitalization(text)
        indicators.proper_punctuation = self._check_punctuation(text)
        indicators.minimal_typos = self._check_spelling_quality(text)
        indicators.coherent_structure = self._check_coherence(text)

        # HTML structure indicators
        if html_content and HAS_BS4:
            await self._analyze_html_structure(html_content, indicators)
        elif html_content:
            await self._analyze_html_structure_regex(html_content, indicators)

        # Calculate completeness scores
        indicators.content_completeness = self._calculate_content_completeness(text)
        indicators.meta_completeness = self._calculate_meta_completeness(indicators)

        return indicators

    async def _analyze_html_structure(self, html_content: str, indicators: QualityIndicators) -> None:
        """Analyze HTML structure using BeautifulSoup."""
        try:
            soup = BeautifulSoup(html_content, "html.parser")

            # Check for structural elements
            indicators.has_title = bool(soup.find("title"))
            indicators.has_headings = bool(soup.find(["h1", "h2", "h3", "h4", "h5", "h6"]))
            indicators.has_paragraphs = bool(soup.find("p"))
            indicators.has_lists = bool(soup.find(["ul", "ol", "li"]))
            indicators.has_links = bool(soup.find("a", href=True))
            indicators.has_images = bool(soup.find("img", src=True))

            # Check for meta elements
            indicators.has_meta_description = bool(
                soup.find("meta", attrs={"name": "description"})
                or soup.find("meta", attrs={"property": "og:description"})
            )

            # Check for structured data
            indicators.has_structured_data = bool(
                soup.find("script", type="application/ld+json") or soup.find(attrs={"itemscope": True})
            )

            # Check for author information
            indicators.has_author = bool(
                soup.find("meta", attrs={"name": "author"})
                or soup.find(attrs={"rel": "author"})
                or soup.select('[class*="author"]')
            )

            # Check for date information
            indicators.has_date = bool(
                soup.find("time")
                or soup.find("meta", attrs={"property": "article:published_time"})
                or soup.select('[class*="date"]')
            )

        except Exception as e:
            logger.warning(f"HTML structure analysis error: {e}")

    async def _analyze_html_structure_regex(self, html_content: str, indicators: QualityIndicators) -> None:
        """Fallback HTML structure analysis using regex."""
        # Check for structural elements
        indicators.has_title = bool(re.search(r"<title[^>]*>", html_content, re.IGNORECASE))
        indicators.has_headings = bool(re.search(r"<h[1-6][^>]*>", html_content, re.IGNORECASE))
        indicators.has_paragraphs = bool(re.search(r"<p[^>]*>", html_content, re.IGNORECASE))
        indicators.has_lists = bool(re.search(r"<[uo]l[^>]*>", html_content, re.IGNORECASE))
        indicators.has_links = bool(re.search(r"<a[^>]*href", html_content, re.IGNORECASE))
        indicators.has_images = bool(re.search(r"<img[^>]*src", html_content, re.IGNORECASE))

        # Check for meta elements
        indicators.has_meta_description = bool(
            re.search(r'<meta[^>]*name=["\']description["\']', html_content, re.IGNORECASE)
            or re.search(
                r'<meta[^>]*property=["\']og:description["\']',
                html_content,
                re.IGNORECASE,
            )
        )

        # Check for structured data
        indicators.has_structured_data = bool(
            re.search(r"application/ld\+json", html_content, re.IGNORECASE)
            or re.search(r"itemscope", html_content, re.IGNORECASE)
        )

        # Check for author information
        indicators.has_author = bool(
            re.search(r'<meta[^>]*name=["\']author["\']', html_content, re.IGNORECASE)
            or re.search(r'rel=["\']author["\']', html_content, re.IGNORECASE)
            or re.search(r'class=["\'][^"\']*author[^"\']*["\']', html_content, re.IGNORECASE)
        )

        # Check for date information
        indicators.has_date = bool(
            re.search(r"<time[^>]*>", html_content, re.IGNORECASE)
            or re.search(r"article:published_time", html_content, re.IGNORECASE)
            or re.search(r'class=["\'][^"\']*date[^"\']*["\']', html_content, re.IGNORECASE)
        )

    async def _categorize_content(self, text: str) -> List[str]:
        """Categorize content based on keyword analysis."""
        categories = []
        text_lower = text.lower()

        # Score each category
        category_scores = {}
        for category, pattern in self.category_patterns.items():
            matches = pattern.findall(text_lower)
            if matches:
                # Score based on frequency and text length
                score = len(matches) / max(len(text.split()), 1) * 1000
                category_scores[category] = score

        # Return categories above threshold, sorted by score
        threshold = 0.1
        categories = [category for category, score in category_scores.items() if score >= threshold]

        # Sort by score (descending)
        categories.sort(key=lambda c: category_scores[c], reverse=True)

        return categories[:3]  # Return top 3 categories

    async def _extract_topics(self, text: str) -> List[str]:
        """Extract key topics from content."""
        # Simple topic extraction based on noun phrases
        topics = []

        try:
            # Find potential topics (capitalized words, noun phrases)
            topic_patterns = [
                r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b",  # Capitalized phrases
                r"\b(?:artificial intelligence|machine learning|data science|cloud computing)\b",  # Known topics
            ]

            for pattern in topic_patterns:
                matches = re.finditer(pattern, text)
                for match in matches:
                    topic = match.group(0).strip()
                    if len(topic) > 3 and topic.lower() not in self.stop_words:
                        topics.append(topic)

            # Count frequency and return most common
            topic_counts = Counter(topics)
            return [topic for topic, count in topic_counts.most_common(5)]

        except Exception as e:
            logger.warning(f"Topic extraction error: {e}")
            return []

    def _count_syllables(self, text: str) -> int:
        """Estimate syllable count in text."""
        words = re.findall(r"\b\w+\b", text.lower())
        total_syllables = 0

        for word in words:
            # Simple syllable counting heuristic
            syllables = len(re.findall(r"[aeiouy]+", word))
            if word.endswith("e"):
                syllables -= 1
            if syllables == 0:
                syllables = 1
            total_syllables += syllables

        return total_syllables

    def _calculate_flesch_reading_ease(self, words: int, sentences: int, syllables: int) -> Optional[float]:
        """Calculate Flesch Reading Ease score."""
        if sentences == 0 or words == 0:
            return None

        score = 206.835 - (1.015 * (words / sentences)) - (84.6 * (syllables / words))
        return max(0, min(100, score))

    def _calculate_flesch_kincaid_grade(self, words: int, sentences: int, syllables: int) -> Optional[float]:
        """Calculate Flesch-Kincaid Grade Level."""
        if sentences == 0 or words == 0:
            return None

        grade = (0.39 * (words / sentences)) + (11.8 * (syllables / words)) - 15.59
        return max(0, grade)

    def _calculate_mattr(self, words: List[str], window_size: int = 100) -> float:
        """Calculate Moving Average Type-Token Ratio."""
        if len(words) < window_size:
            return len(set(words)) / len(words)

        ttrs = []
        for i in range(len(words) - window_size + 1):
            window = words[i : i + window_size]
            ttr = len(set(window)) / len(window)
            ttrs.append(ttr)

        return sum(ttrs) / len(ttrs)

    def _count_complex_words(self, words: List[str]) -> int:
        """Count words with 3 or more syllables."""
        complex_count = 0

        for word in words:
            syllables = len(re.findall(r"[aeiouy]+", word.lower()))
            if word.lower().endswith("e"):
                syllables -= 1
            if syllables == 0:
                syllables = 1

            if syllables >= 3:
                complex_count += 1

        return complex_count

    def _check_capitalization(self, text: str) -> bool:
        """Check if text has proper capitalization."""
        sentences = re.split(r"[.!?]+", text)
        proper_sentences = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and sentence[0].isupper():
                proper_sentences += 1

        return len(sentences) > 0 and proper_sentences / len(sentences) > 0.8

    def _check_punctuation(self, text: str) -> bool:
        """Check if text has proper punctuation."""
        # Check for basic punctuation patterns
        has_periods = "." in text
        has_sentence_endings = bool(re.search(r"[.!?]", text))

        # Check that sentences end with punctuation
        sentences = text.split(".")
        if len(sentences) > 1:
            return has_sentence_endings

        return has_periods

    def _check_spelling_quality(self, text: str) -> bool:
        """Basic spelling quality check."""
        # Look for common spelling error patterns
        error_patterns = [
            r"\b\w*\w\w\w\w+\b",  # Very long words (potential typos)
            r"\b\w*[0-9]+\w*\b",  # Words with numbers mixed in
            r"\b[a-z][A-Z]",  # Inconsistent capitalization within words
        ]

        error_count = 0
        for pattern in error_patterns:
            error_count += len(re.findall(pattern, text))

        word_count = len(text.split())
        if word_count == 0:
            return False

        error_rate = error_count / word_count
        return error_rate < 0.05  # Less than 5% error rate

    def _check_coherence(self, text: str) -> bool:
        """Basic coherence check."""
        # Check for reasonable sentence length distribution
        sentences = re.split(r"[.!?]+", text)
        sentence_lengths = [len(s.split()) for s in sentences if s.strip()]

        if not sentence_lengths:
            return False

        avg_length = sum(sentence_lengths) / len(sentence_lengths)

        # Good coherence: average sentence length between 10-25 words
        return 10 <= avg_length <= 25

    def _calculate_content_completeness(self, text: str) -> float:
        """Calculate content completeness score."""
        score = 0.0

        # Length factor (longer content generally more complete)
        word_count = len(text.split())
        if word_count >= 300:
            score += 0.3
        elif word_count >= 100:
            score += 0.2
        elif word_count >= 50:
            score += 0.1

        # Structure factor
        sentences = len(re.split(r"[.!?]+", text))
        if sentences >= 5:
            score += 0.2
        elif sentences >= 3:
            score += 0.1

        # Paragraph factor
        paragraphs = len([p for p in text.split("\n\n") if p.strip()])
        if paragraphs >= 3:
            score += 0.2
        elif paragraphs >= 2:
            score += 0.1

        # Content depth (presence of detailed information)
        if any(word in text.lower() for word in ["because", "therefore", "however", "although"]):
            score += 0.2

        # Informational content
        if any(word in text.lower() for word in ["study", "research", "analysis", "data", "results"]):
            score += 0.1

        return min(1.0, score)

    def _calculate_meta_completeness(self, indicators: QualityIndicators) -> float:
        """Calculate metadata completeness score."""
        meta_factors = [
            indicators.has_title,
            indicators.has_author,
            indicators.has_date,
            indicators.has_meta_description,
            indicators.has_structured_data,
        ]

        return sum(meta_factors) / len(meta_factors)

    def _calculate_quality_score(self, metrics: ContentMetrics) -> float:
        """Calculate overall quality score."""
        score = 0.0

        # Reading metrics contribution (30%)
        if metrics.reading_metrics.word_count >= 100:
            score += 0.1
        if metrics.reading_metrics.flesch_reading_ease:
            # Prefer readability scores between 30-70
            ease = metrics.reading_metrics.flesch_reading_ease
            if 30 <= ease <= 70:
                score += 0.2
            elif 20 <= ease <= 80:
                score += 0.1

        # Lexical diversity contribution (25%)
        if metrics.lexical_metrics.lexical_diversity > 0.3:
            score += 0.15
        elif metrics.lexical_metrics.lexical_diversity > 0.2:
            score += 0.1

        if metrics.lexical_metrics.vocabulary_richness > 0.7:
            score += 0.1
        elif metrics.lexical_metrics.vocabulary_richness > 0.5:
            score += 0.05

        # Quality indicators contribution (45%)
        structure_score = (
            sum(
                [
                    metrics.quality_indicators.has_title,
                    metrics.quality_indicators.has_headings,
                    metrics.quality_indicators.has_paragraphs,
                    metrics.quality_indicators.proper_capitalization,
                    metrics.quality_indicators.proper_punctuation,
                    metrics.quality_indicators.minimal_typos,
                    metrics.quality_indicators.coherent_structure,
                ]
            )
            / 7
            * 0.25
        )

        meta_score = metrics.quality_indicators.meta_completeness * 0.1
        content_score = metrics.quality_indicators.content_completeness * 0.1

        score += structure_score + meta_score + content_score

        return min(1.0, score)

    def _calculate_readability_score(self, metrics: ContentMetrics) -> float:
        """Calculate readability score."""
        if not metrics.reading_metrics.flesch_reading_ease:
            return 0.5  # Default neutral score

        # Convert Flesch Reading Ease to 0-1 scale
        ease = metrics.reading_metrics.flesch_reading_ease

        # Optimal range is 30-70, with 50 being ideal
        if 40 <= ease <= 60:
            return 1.0
        elif 30 <= ease <= 70:
            return 0.8
        elif 20 <= ease <= 80:
            return 0.6
        elif 10 <= ease <= 90:
            return 0.4
        else:
            return 0.2

    def _calculate_engagement_score(self, metrics: ContentMetrics) -> float:
        """Calculate engagement potential score."""
        score = 0.0

        # Length factor (optimal length for engagement)
        word_count = metrics.word_count
        if 300 <= word_count <= 1500:
            score += 0.3
        elif 150 <= word_count <= 2500:
            score += 0.2
        elif word_count >= 100:
            score += 0.1

        # Reading time factor (optimal 2-8 minutes)
        reading_time = metrics.reading_time_minutes
        if 2 <= reading_time <= 8:
            score += 0.2
        elif 1 <= reading_time <= 12:
            score += 0.1

        # Structure factor
        if metrics.quality_indicators.has_headings:
            score += 0.1
        if metrics.quality_indicators.has_lists:
            score += 0.1
        if metrics.quality_indicators.has_images:
            score += 0.1

        # Readability factor
        score += metrics.readability_score * 0.2

        return min(1.0, score)
