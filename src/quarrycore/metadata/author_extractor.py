"""
Author Extractor - Multi-Strategy Author Identification

Extracts author information using structured data, meta tags, bylines,
and spaCy NER for comprehensive author identification in content analysis.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Pattern

# BeautifulSoup imports with graceful fallbacks (proven pattern)
try:
    from bs4 import BeautifulSoup, NavigableString, Tag

    HAS_BS4 = True
except ImportError:
    if TYPE_CHECKING:
        from bs4 import BeautifulSoup, NavigableString, Tag
    else:
        BeautifulSoup = None
        Tag = None
        NavigableString = None
    HAS_BS4 = False

# spaCy imports with graceful fallbacks (proven pattern)
try:
    import spacy
    from spacy import Language

    HAS_SPACY = True
except ImportError:
    if TYPE_CHECKING:
        import spacy
        from spacy import Language
    else:
        spacy = None
        Language = None
    HAS_SPACY = False

logger = logging.getLogger(__name__)


class AuthorConfidence(Enum):
    """Confidence levels for author extraction."""

    HIGH = "high"  # 0.8-1.0: Structured data, clear bylines
    MEDIUM = "medium"  # 0.5-0.8: Meta tags, pattern matching
    LOW = "low"  # 0.2-0.5: NER, heuristic matching
    VERY_LOW = "very_low"  # 0.0-0.2: Weak signals, uncertain


@dataclass
class AuthorInfo:
    """Information about an extracted author."""

    name: str
    confidence: AuthorConfidence
    confidence_score: float
    extraction_method: str

    # Additional author details (when available)
    email: Optional[str] = None
    url: Optional[str] = None
    bio: Optional[str] = None
    social_profiles: Dict[str, str] = field(default_factory=dict)

    # Context information
    source_text: Optional[str] = None
    position_in_content: Optional[int] = None

    def __post_init__(self) -> None:
        """Validate and clean author information."""
        self.name = self.name.strip()

        # Remove common prefixes/suffixes
        prefixes = ["by", "author:", "written by", "posted by", "@"]
        suffixes = ["writes:", "says:", "reports:"]

        name_lower = self.name.lower()
        for prefix in prefixes:
            if name_lower.startswith(prefix):
                self.name = self.name[len(prefix) :].strip()
                break

        for suffix in suffixes:
            if name_lower.endswith(suffix):
                self.name = self.name[: -len(suffix)].strip()
                break

        # Clean up whitespace and punctuation
        self.name = re.sub(r"\s+", " ", self.name)
        self.name = self.name.strip(".,;:")


class AuthorExtractor:
    """
    Multi-strategy author extraction system.

    Uses structured data, meta tags, content analysis, and NLP
    to identify authors with confidence scoring.
    """

    def __init__(self, use_spacy: bool = True) -> None:
        self.use_spacy = use_spacy and HAS_SPACY
        self.nlp: Optional[Any] = None  # spaCy Language object

        # Common author patterns
        self.byline_patterns: List[str] = [
            r"(?:by|author|written by|posted by)\s*:?\s*([a-zA-Z\s\-\.\']+)",
            r"([a-zA-Z\s\-\.\']+)\s*(?:writes|reports|says)",
            r"@([a-zA-Z0-9_]+)",  # Twitter handles
            r"(?:author|writer|journalist):\s*([a-zA-Z\s\-\.\']+)",
        ]

        # Compile patterns
        self.compiled_patterns: List[Pattern[str]] = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.byline_patterns
        ]

        # Author-related CSS selectors (common patterns)
        self.author_selectors: List[str] = [
            '[class*="author"]',
            '[class*="byline"]',
            '[class*="writer"]',
            '[id*="author"]',
            '[rel="author"]',
            ".post-author",
            ".article-author",
            ".by-author",
            ".author-name",
            ".author-info",
            ".byline-author",
        ]

        # Load spaCy model if available
        if self.use_spacy:
            self._load_spacy_model()

        logger.info(f"AuthorExtractor initialized (spaCy: {self.use_spacy})")

    def _load_spacy_model(self) -> None:
        """Load spaCy model for NER."""
        if not HAS_SPACY:
            return

        try:
            # Try to load English model
            model_names = ["en_core_web_sm", "en_core_web_md", "en_core_web_lg"]

            for model_name in model_names:
                try:
                    self.nlp = spacy.load(model_name)
                    logger.info(f"Loaded spaCy model: {model_name}")
                    return
                except OSError:
                    continue

            # If no model found, disable spaCy
            logger.warning("No spaCy English model found, disabling NER")
            self.use_spacy = False

        except Exception as e:
            logger.warning(f"Failed to load spaCy model: {e}")
            self.use_spacy = False

    async def extract_authors(
        self,
        html_content: str,
        text_content: str = "",
        use_nlp: bool = True,
    ) -> List[AuthorInfo]:
        """
        Extract authors using multiple strategies.

        Args:
            html_content: HTML content to analyze
            text_content: Plain text content (optional)
            use_nlp: Whether to use NLP models for extraction

        Returns:
            List of AuthorInfo objects sorted by confidence
        """
        authors: List[AuthorInfo] = []

        # Strategy 1: Structured data extraction
        structured_authors = await self._extract_from_structured_data(html_content)
        authors.extend(structured_authors)

        # Strategy 2: HTML meta tags and attributes
        meta_authors = await self._extract_from_meta_tags(html_content)
        authors.extend(meta_authors)

        # Strategy 3: CSS selector-based extraction
        if HAS_BS4:
            css_authors = await self._extract_from_css_selectors(html_content)
            authors.extend(css_authors)

        # Strategy 4: Pattern matching in content
        pattern_authors = await self._extract_from_patterns(html_content, text_content)
        authors.extend(pattern_authors)

        # Strategy 5: NER-based extraction
        if use_nlp and self.use_spacy and text_content:
            ner_authors = await self._extract_from_ner(text_content)
            authors.extend(ner_authors)

        # Deduplicate and rank authors
        unique_authors = self._deduplicate_authors(authors)

        # Sort by confidence score (descending)
        unique_authors.sort(key=lambda x: x.confidence_score, reverse=True)

        return unique_authors

    async def _extract_from_structured_data(self, html_content: str) -> List[AuthorInfo]:
        """Extract authors from structured data (JSON-LD, microdata)."""
        authors: List[AuthorInfo] = []

        try:
            # Extract JSON-LD data
            json_ld_pattern = r'<script[^>]*type=["\']application/ld\+json["\'][^>]*>(.*?)</script>'
            json_ld_matches = re.finditer(json_ld_pattern, html_content, re.IGNORECASE | re.DOTALL)

            for match in json_ld_matches:
                try:
                    import json

                    json_text = match.group(1).strip()
                    data = json.loads(json_text)

                    # Handle both single objects and arrays
                    items = data if isinstance(data, list) else [data]

                    for item in items:
                        author_data = item.get("author")
                        if author_data:
                            author_names = self._extract_author_names_from_schema(author_data)
                            for name in author_names:
                                authors.append(
                                    AuthorInfo(
                                        name=name,
                                        confidence=AuthorConfidence.HIGH,
                                        confidence_score=0.9,
                                        extraction_method="json_ld_schema",
                                    )
                                )

                except json.JSONDecodeError:
                    continue

        except Exception as e:
            logger.warning(f"Structured data author extraction error: {e}")

        return authors

    def _extract_author_names_from_schema(self, author_data: Any) -> List[str]:
        """Extract author names from Schema.org author data."""
        names: List[str] = []

        try:
            if isinstance(author_data, str):
                names.append(author_data)
            elif isinstance(author_data, dict):
                # Person or Organization schema
                name = author_data.get("name")
                if name:
                    names.append(name)
            elif isinstance(author_data, list):
                # Multiple authors
                for author in author_data:
                    if isinstance(author, str):
                        names.append(author)
                    elif isinstance(author, dict) and author.get("name"):
                        names.append(author["name"])

        except Exception as e:
            logger.warning(f"Schema author name extraction error: {e}")

        return names

    async def _extract_from_meta_tags(self, html_content: str) -> List[AuthorInfo]:
        """Extract authors from HTML meta tags."""
        authors: List[AuthorInfo] = []

        try:
            # Pattern for meta tags
            meta_patterns = [
                r'<meta\s+name=["\']author["\'][^>]*content=["\']([^"\']+)["\']',
                r'<meta\s+content=["\']([^"\']+)["\'][^>]*name=["\']author["\']',
                r'<meta\s+property=["\']article:author["\'][^>]*content=["\']([^"\']+)["\']',
                r'<meta\s+name=["\']dc\.creator["\'][^>]*content=["\']([^"\']+)["\']',
            ]

            for pattern in meta_patterns:
                matches = re.finditer(pattern, html_content, re.IGNORECASE)
                for match in matches:
                    author_name = match.group(1).strip()
                    if author_name and len(author_name) > 1:
                        authors.append(
                            AuthorInfo(
                                name=author_name,
                                confidence=AuthorConfidence.MEDIUM,
                                confidence_score=0.7,
                                extraction_method="meta_tag",
                            )
                        )

        except Exception as e:
            logger.warning(f"Meta tag author extraction error: {e}")

        return authors

    async def _extract_from_css_selectors(self, html_content: str) -> List[AuthorInfo]:
        """Extract authors using CSS selectors on common author elements."""
        authors: List[AuthorInfo] = []

        if not HAS_BS4:
            return authors

        try:
            soup = BeautifulSoup(html_content, "html.parser")

            for selector in self.author_selectors:
                elements = soup.select(selector)

                for element in elements:
                    # Get text content
                    text = element.get_text(strip=True)

                    # Check for links (author profiles) - handle Union type properly
                    author_url: Optional[str] = None
                    if HAS_BS4 and hasattr(element, "find") and hasattr(element, "name"):  # Tag element
                        link = element.find("a")
                        if link and hasattr(link, "get") and hasattr(link, "name"):  # Also a Tag
                            href = link.get("href")
                            if isinstance(href, str):
                                author_url = href
                            elif isinstance(href, list) and href:
                                author_url = href[0]  # Take first URL if multiple

                    # Clean text to extract just the author name
                    cleaned_text = self._clean_author_text_from_css(text)

                    # Clean and validate author name
                    if cleaned_text and len(cleaned_text) > 1 and len(cleaned_text) < 100:
                        # Skip if it looks like a date or number
                        if not re.match(r"^\d+[\d\s\-/]+$", cleaned_text):
                            authors.append(
                                AuthorInfo(
                                    name=cleaned_text,
                                    confidence=AuthorConfidence.MEDIUM,
                                    confidence_score=0.6,
                                    extraction_method="css_selector",
                                    url=author_url,
                                    source_text=text,
                                )
                            )

        except Exception as e:
            logger.warning(f"CSS selector author extraction error: {e}")

        return authors

    async def _extract_from_patterns(self, html_content: str, text_content: str) -> List[AuthorInfo]:
        """Extract authors using regex patterns."""
        authors: List[AuthorInfo] = []

        try:
            # Combine HTML and text content for pattern matching
            content_to_search = f"{html_content}\n{text_content}"

            for i, pattern in enumerate(self.compiled_patterns):
                matches = pattern.finditer(content_to_search)

                for match in matches:
                    author_name = match.group(1).strip()

                    # Validate author name
                    if self._is_valid_author_name(author_name):
                        # Calculate confidence based on pattern type and position
                        confidence_score = 0.5 - (i * 0.1)  # Earlier patterns have higher confidence

                        # Boost confidence if found early in content
                        if match.start() < len(content_to_search) * 0.2:
                            confidence_score += 0.1

                        authors.append(
                            AuthorInfo(
                                name=author_name,
                                confidence=(
                                    AuthorConfidence.MEDIUM if confidence_score > 0.5 else AuthorConfidence.LOW
                                ),
                                confidence_score=max(0.2, confidence_score),
                                extraction_method="pattern_matching",
                                source_text=match.group(0),
                                position_in_content=match.start(),
                            )
                        )

        except Exception as e:
            logger.warning(f"Pattern-based author extraction error: {e}")

        return authors

    async def _extract_from_ner(self, text_content: str) -> List[AuthorInfo]:
        """Extract authors using spaCy Named Entity Recognition."""
        authors: List[AuthorInfo] = []

        if not self.use_spacy or not self.nlp:
            return authors

        try:
            # Process text with spaCy
            doc = self.nlp(text_content[:10000])  # Limit text length for performance

            # Look for PERSON entities
            person_entities = [ent for ent in doc.ents if ent.label_ == "PERSON"]

            # Score entities based on context
            for ent in person_entities:
                author_name = ent.text.strip()

                if self._is_valid_author_name(author_name):
                    # Calculate confidence based on context
                    confidence_score = self._calculate_ner_confidence(ent, doc)

                    authors.append(
                        AuthorInfo(
                            name=author_name,
                            confidence=self._score_to_confidence_level(confidence_score),
                            confidence_score=confidence_score,
                            extraction_method="spacy_ner",
                            source_text=ent.sent.text if ent.sent else None,
                            position_in_content=ent.start_char,
                        )
                    )

        except Exception as e:
            logger.warning(f"NER author extraction error: {e}")

        return authors

    def _is_valid_author_name(self, name: str) -> bool:
        """Validate if a string looks like a valid author name."""
        if not name or len(name) < 2 or len(name) > 100:
            return False

        # Skip common non-author patterns
        invalid_patterns = [
            r"^\d+$",  # Just numbers
            r"^[A-Z]{2,}$",  # All caps (likely acronyms)
            r"^\w+@\w+\.\w+$",  # Email addresses
            r"^https?://",  # URLs
            r"^\d{1,2}[/\-]\d{1,2}",  # Dates
            r"^(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)",  # Month names
            r"^(monday|tuesday|wednesday|thursday|friday|saturday|sunday)",  # Day names
        ]

        name_lower = name.lower()
        for pattern in invalid_patterns:
            if re.match(pattern, name_lower):
                return False

        # Must contain at least one letter
        if not re.search(r"[a-zA-Z]", name):
            return False

        # Skip very common words that aren't names
        common_words = {
            "admin",
            "administrator",
            "author",
            "editor",
            "writer",
            "staff",
            "team",
            "news",
            "press",
            "media",
            "content",
            "article",
            "post",
            "blog",
            "website",
            "page",
            "home",
            "about",
            "contact",
            "privacy",
        }

        if name_lower in common_words:
            return False

        return True

    def _calculate_ner_confidence(self, entity: Any, doc: Any) -> float:
        """Calculate confidence score for NER-extracted entity."""
        base_confidence = 0.4

        # Boost confidence based on context
        sentence = entity.sent
        if sentence:
            sentence_text = sentence.text.lower()

            # Look for author-indicating context
            author_indicators = [
                "by",
                "author",
                "written",
                "posted",
                "says",
                "reports",
                "writes",
            ]
            for indicator in author_indicators:
                if indicator in sentence_text:
                    base_confidence += 0.2
                    break

        # Boost if entity appears early in text
        if entity.start_char < len(doc.text) * 0.1:
            base_confidence += 0.1

        # Penalize if entity appears very late
        if entity.start_char > len(doc.text) * 0.8:
            base_confidence -= 0.1

        return min(0.8, max(0.1, base_confidence))

    def _score_to_confidence_level(self, score: float) -> AuthorConfidence:
        """Convert numeric confidence score to confidence level."""
        if score >= 0.8:
            return AuthorConfidence.HIGH
        elif score >= 0.5:
            return AuthorConfidence.MEDIUM
        elif score >= 0.2:
            return AuthorConfidence.LOW
        else:
            return AuthorConfidence.VERY_LOW

    def _deduplicate_authors(self, authors: List[AuthorInfo]) -> List[AuthorInfo]:
        """Remove duplicate authors, keeping the highest confidence version."""
        if not authors:
            return []

        # Group by normalized name
        author_groups: Dict[str, List[AuthorInfo]] = {}

        for author in authors:
            # Normalize name for comparison
            normalized_name = self._normalize_name(author.name)

            if normalized_name not in author_groups:
                author_groups[normalized_name] = []
            author_groups[normalized_name].append(author)

        # Keep best author from each group
        unique_authors: List[AuthorInfo] = []
        for group in author_groups.values():
            # Sort by confidence score and take the best one
            best_author = max(group, key=lambda x: x.confidence_score)

            # Merge information from other extractions if available
            for other_author in group:
                if other_author != best_author:
                    # Merge URLs if missing
                    if not best_author.url and other_author.url:
                        best_author.url = other_author.url

                    # Merge social profiles
                    best_author.social_profiles.update(other_author.social_profiles)

                    # Use the most detailed bio
                    if not best_author.bio and other_author.bio:
                        best_author.bio = other_author.bio

            unique_authors.append(best_author)

        return unique_authors

    def _normalize_name(self, name: str) -> str:
        """Normalize author name for comparison with enhanced fuzzy matching."""
        # Convert to lowercase and remove extra whitespace
        normalized = re.sub(r"\s+", " ", name.lower().strip())

        # Remove common punctuation
        normalized = re.sub(r'[.,;:\'"()]', "", normalized)

        # Remove common prefixes/suffixes for comparison
        prefixes = [
            "by ",
            "author ",
            "dr ",
            "mr ",
            "ms ",
            "mrs ",
            "prof ",
            "professor ",
        ]
        for prefix in prefixes:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix) :].strip()
                break

        # Remove common suffixes (degrees, titles, etc.)
        suffixes = [
            "phd",
            "ph.d",
            "md",
            "m.d",
            "jr",
            "sr",
            "ii",
            "iii",
            "iv",
            "phd in computer science",
            "phd in data science",
            "phd in ai",
            "computer scientist",
            "data scientist",
            "researcher",
        ]
        for suffix in suffixes:
            if normalized.endswith(" " + suffix):
                normalized = normalized[: -len(" " + suffix)].strip()
            elif normalized.endswith(suffix):
                normalized = normalized[: -len(suffix)].strip()

        # Extract core name (first and last name typically)
        # Remove extra descriptive text after commas
        if "," in normalized:
            normalized = normalized.split(",")[0].strip()

        return normalized

    def _clean_author_text_from_css(self, text: str) -> str:
        """Clean author text extracted from CSS selectors to remove dates and extra content."""
        if not text:
            return ""

        # Remove common date patterns that might be concatenated with author names
        # Pattern: "Dr. Jane SmithDecember 1, 2023" -> "Dr. Jane Smith"
        date_patterns = [
            r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}",
            r"\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}",
            r"\d{4}[/\-]\d{1,2}[/\-]\d{1,2}",
            r"\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}",
        ]

        cleaned = text
        for pattern in date_patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

        # Remove time patterns like "10:00 AM" or "15:30"
        time_patterns = [
            r"\d{1,2}:\d{2}\s*(?:AM|PM)?",
            r"\d{1,2}:\d{2}:\d{2}",
        ]

        for pattern in time_patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

        # Clean up extra whitespace and punctuation
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        cleaned = cleaned.strip(".,;:-")

        return cleaned

    async def extract_author_details(self, author_name: str, html_content: str) -> Dict[str, Any]:
        """
        Extract detailed information about a specific author.

        Args:
            author_name: Name of the author to look up
            html_content: HTML content to search for author details

        Returns:
            Dictionary with detailed author information
        """
        details: Dict[str, Any] = {
            "name": author_name,
            "bio": None,
            "email": None,
            "social_profiles": {},
            "profile_urls": [],
            "image_url": None,
            "confidence": 0.0,
        }

        if not HAS_BS4:
            return details

        try:
            soup = BeautifulSoup(html_content, "html.parser")

            # Look for author bio/description
            bio_selectors = [
                f'[class*="author"][class*="bio"]:contains("{author_name}")',
                f'[class*="author"][class*="description"]:contains("{author_name}")',
                f'[class*="bio"]:contains("{author_name}")',
            ]

            for selector in bio_selectors:
                try:
                    elements = soup.select(selector)
                    if elements:
                        bio_text = elements[0].get_text(strip=True)
                        if len(bio_text) > 20:  # Reasonable bio length
                            details["bio"] = bio_text
                            break
                except Exception:
                    continue

            # Look for author links and social profiles
            author_links = soup.find_all("a", href=True)
            for link in author_links:
                href = link.get("href", "")
                link_text = link.get_text(strip=True).lower()

                # Check if link is related to this author
                if author_name.lower() in link_text or author_name.lower() in href.lower():
                    details["profile_urls"].append(href)

                    # Extract social media profiles
                    if "twitter.com" in href:
                        details["social_profiles"]["twitter"] = href
                    elif "linkedin.com" in href:
                        details["social_profiles"]["linkedin"] = href
                    elif "facebook.com" in href:
                        details["social_profiles"]["facebook"] = href
                    elif "instagram.com" in href:
                        details["social_profiles"]["instagram"] = href

            # Look for author email
            email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
            # Search in context around author name
            author_context = self._get_author_context(author_name, html_content)
            email_matches = re.finditer(email_pattern, author_context)
            for match in email_matches:
                details["email"] = match.group(0)
                break

            # Calculate confidence based on available information
            confidence_factors = [
                bool(details["bio"]),
                bool(details["email"]),
                bool(details["social_profiles"]),
                bool(details["profile_urls"]),
            ]
            details["confidence"] = sum(confidence_factors) / len(confidence_factors)

        except Exception as e:
            logger.warning(f"Author details extraction error: {e}")

        return details

    def _get_author_context(self, author_name: str, html_content: str, context_size: int = 500) -> str:
        """Get text context around author name mentions."""
        context_parts = []

        # Find all mentions of the author name
        pattern = re.compile(re.escape(author_name), re.IGNORECASE)
        matches = pattern.finditer(html_content)

        for match in matches:
            start = max(0, match.start() - context_size)
            end = min(len(html_content), match.end() + context_size)
            context_parts.append(html_content[start:end])

        return " ".join(context_parts)
