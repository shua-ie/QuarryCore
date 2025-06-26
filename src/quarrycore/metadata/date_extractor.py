"""
Date Extractor - Multi-Strategy Publication Date Detection

Extracts publication dates using structured data, meta tags, URL patterns,
and content analysis for comprehensive temporal metadata extraction.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

# Optional dependencies handling
try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False
    BeautifulSoup = None  # type: ignore

try:
    from dateutil import parser as dateutil_parser
    HAS_DATEUTIL = True
except ImportError:
    HAS_DATEUTIL = False
    dateutil_parser = None  # type: ignore

logger = logging.getLogger(__name__)


class DateExtractionStrategy(Enum):
    """Date extraction strategies ordered by reliability."""
    STRUCTURED_DATA = "structured_data"      # JSON-LD, microdata
    META_TAGS = "meta_tags"                  # HTML meta tags
    URL_PATTERN = "url_pattern"              # Date patterns in URL
    TIME_ELEMENT = "time_element"            # HTML time elements
    CONTENT_PATTERN = "content_pattern"      # Date patterns in content
    FILENAME_PATTERN = "filename_pattern"    # Date in filename/path
    HEURISTIC = "heuristic"                  # Best guess from context


@dataclass
class DateInfo:
    """Information about an extracted publication date."""
    
    date: datetime
    confidence: float
    extraction_method: DateExtractionStrategy
    source_text: Optional[str] = None
    
    # Additional temporal information
    is_approximate: bool = False
    date_format: Optional[str] = None
    timezone: Optional[str] = None
    
    # Context information
    element_tag: Optional[str] = None
    element_attributes: Optional[Dict[str, str]] = None


class DateExtractor:
    """
    Multi-strategy publication date extraction system.
    
    Uses structured data, meta tags, URL analysis, and content patterns
    to identify publication dates with confidence scoring.
    """
    
    def __init__(self) -> None:
        # Common date patterns (ordered by reliability)
        self.date_patterns = [
            # ISO 8601 formats
            (r'\b(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?)\b', 'iso_datetime'),
            (r'\b(\d{4}-\d{2}-\d{2})\b', 'iso_date'),
            
            # Common formats
            (r'\b(\d{1,2}/\d{1,2}/\d{4})\b', 'us_date'),
            (r'\b(\d{1,2}-\d{1,2}-\d{4})\b', 'dash_date'),
            (r'\b(\d{1,2}\.\d{1,2}\.\d{4})\b', 'dot_date'),
            
            # Month name formats
            (r'\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4})\b', 'month_name'),
            (r'\b(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4})\b', 'day_month_year'),
            
            # Timestamp formats
            (r'\b(\d{10})\b', 'unix_timestamp'),
            (r'\b(\d{13})\b', 'unix_timestamp_ms'),
        ]
        
        # Compile regex patterns
        self.compiled_patterns = [(re.compile(pattern, re.IGNORECASE), format_type) 
                                 for pattern, format_type in self.date_patterns]
        
        # URL date patterns
        self.url_date_patterns = [
            r'/(\d{4})/(\d{1,2})/(\d{1,2})/',      # /2023/12/25/
            r'/(\d{4})-(\d{1,2})-(\d{1,2})/',      # /2023-12-25/
            r'/(\d{4})(\d{2})(\d{2})/',            # /20231225/
            r'_(\d{4})(\d{2})(\d{2})_',            # _20231225_
            r'-(\d{4})-(\d{1,2})-(\d{1,2})-',      # -2023-12-25-
        ]
        
        # Compile URL patterns
        self.compiled_url_patterns = [re.compile(pattern) for pattern in self.url_date_patterns]
        
        # Meta tag date selectors
        self.meta_date_selectors = [
            ('meta[property="article:published_time"]', 'content'),
            ('meta[property="article:modified_time"]', 'content'),
            ('meta[name="publish_date"]', 'content'),
            ('meta[name="publication_date"]', 'content'),
            ('meta[name="date"]', 'content'),
            ('meta[name="DC.date"]', 'content'),
            ('meta[name="DC.date.created"]', 'content'),
            ('meta[name="DC.date.issued"]', 'content'),
            ('meta[name="sailthru.date"]', 'content'),
            ('meta[name="article.published"]', 'content'),
            ('meta[name="published-date"]', 'content'),
            ('meta[name="release_date"]', 'content'),
            ('meta[name="created"]', 'content'),
        ]
        
        # Time element selectors
        self.time_selectors = [
            'time[datetime]',
            'time[pubdate]',
            '.published',
            '.date-published',
            '.publish-date',
            '.article-date',
            '.post-date',
            '.entry-date',
            '.timestamp',
        ]
        
        logger.info("DateExtractor initialized with multi-strategy detection")
    
    async def extract_publication_date(
        self,
        html_content: str,
        url: str = "",
        text_content: str = "",
    ) -> Optional[DateInfo]:
        """
        Extract publication date using multiple strategies.
        
        Args:
            html_content: HTML content to analyze
            url: URL of the content (for URL pattern analysis)
            text_content: Plain text content (optional)
            
        Returns:
            DateInfo object with best date found, or None
        """
        candidates: List[DateInfo] = []
        
        # Strategy 1: Structured data (highest confidence)
        structured_dates = await self._extract_from_structured_data(html_content)
        candidates.extend(structured_dates)
        
        # Strategy 2: Meta tags (high confidence)
        meta_dates = await self._extract_from_meta_tags(html_content)
        candidates.extend(meta_dates)
        
        # Strategy 3: Time elements (medium confidence)
        if HAS_BS4:
            time_dates = await self._extract_from_time_elements(html_content)
            candidates.extend(time_dates)
        
        # Strategy 4: URL patterns (medium confidence)
        if url:
            url_dates = await self._extract_from_url(url)
            candidates.extend(url_dates)
        
        # Strategy 5: Content patterns (lower confidence)
        content_dates = await self._extract_from_content_patterns(html_content, text_content)
        candidates.extend(content_dates)
        
        # Select best candidate
        if candidates:
            # Sort by confidence score (descending)
            candidates.sort(key=lambda x: x.confidence, reverse=True)
            
            # Filter out obviously wrong dates
            valid_candidates = [c for c in candidates if self._is_reasonable_date(c.date)]
            
            if valid_candidates:
                return valid_candidates[0]
        
        return None
    
    async def _extract_from_structured_data(self, html_content: str) -> List[DateInfo]:
        """Extract dates from JSON-LD and microdata."""
        dates: List[DateInfo] = []
        
        try:
            # Extract JSON-LD data
            json_ld_pattern = r'<script[^>]*type=["\']application/ld\+json["\'][^>]*>(.*?)</script>'
            json_ld_matches = re.finditer(json_ld_pattern, html_content, re.IGNORECASE | re.DOTALL)
            
            for match in json_ld_matches:
                try:
                    json_text = match.group(1).strip()
                    data = json.loads(json_text)
                    
                    # Handle both single objects and arrays
                    items = data if isinstance(data, list) else [data]
                    
                    for item in items:
                        # Look for date fields
                        date_fields = [
                            'datePublished', 'dateCreated', 'dateModified',
                            'publishDate', 'publicationDate', 'created',
                            'modified', 'uploadDate'
                        ]
                        
                        for field in date_fields:
                            date_value = item.get(field)
                            if date_value:
                                parsed_date = self._parse_date_string(str(date_value))
                                if parsed_date:
                                    confidence = 0.95 if field in ['datePublished', 'publishDate'] else 0.85
                                    dates.append(DateInfo(
                                        date=parsed_date,
                                        confidence=confidence,
                                        extraction_method=DateExtractionStrategy.STRUCTURED_DATA,
                                        source_text=str(date_value),
                                        date_format='json_ld',
                                    ))
                
                except json.JSONDecodeError:
                    continue
        
        except Exception as e:
            logger.warning(f"Structured data date extraction error: {e}")
        
        return dates
    
    async def _extract_from_meta_tags(self, html_content: str) -> List[DateInfo]:
        """Extract dates from HTML meta tags."""
        dates: List[DateInfo] = []
        
        try:
            if HAS_BS4:
                soup = BeautifulSoup(html_content, 'html.parser')
                
                for selector, attribute in self.meta_date_selectors:
                    elements = soup.select(selector)
                    for element in elements:
                        if not element:
                            continue
                        date_value = element.get(attribute)
                        if isinstance(date_value, list):
                            date_value = date_value[0] if date_value else None

                        if date_value:
                            parsed_date = self._parse_date_string(str(date_value))
                            if parsed_date:
                                # Higher confidence for article-specific tags
                                confidence = 0.9 if 'article' in selector else 0.8
                                dates.append(DateInfo(
                                    date=parsed_date,
                                    confidence=confidence,
                                    extraction_method=DateExtractionStrategy.META_TAGS,
                                    source_text=date_value,
                                    element_tag=element.name,
                                    element_attributes=dict(element.attrs),
                                ))
            else:
                # Fallback regex parsing
                for selector, attribute in self.meta_date_selectors:
                    # Convert CSS selector to regex pattern
                    if 'property=' in selector:
                        property_match = re.search(r'property="([^"]+)"', selector)
                        if property_match:
                            property_name = property_match.group(1)
                            pattern = f'<meta[^>]*property=["\'][^"\']*{re.escape(property_name)}[^"\']*["\'][^>]*content=["\']([^"\']+)["\']'
                            matches = re.finditer(pattern, html_content, re.IGNORECASE)
                            for match in matches:
                                date_value = match.group(1)
                                parsed_date = self._parse_date_string(str(date_value))
                                if parsed_date:
                                    dates.append(DateInfo(
                                        date=parsed_date,
                                        confidence=0.8,
                                        extraction_method=DateExtractionStrategy.META_TAGS,
                                        source_text=date_value,
                                    ))
        
        except Exception as e:
            logger.warning(f"Meta tag date extraction error: {e}")
        
        return dates
    
    async def _extract_from_time_elements(self, html_content: str) -> List[DateInfo]:
        """Extract dates from HTML time elements."""
        dates: List[DateInfo] = []
        
        if not HAS_BS4:
            return dates
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            for selector in self.time_selectors:
                elements = soup.select(selector)
                
                for element in elements:
                    if not element:
                        continue
                    date_value = None
                    
                    # Try different attributes
                    for attr in ['datetime', 'title', 'data-time', 'data-date']:
                        date_value = element.get(attr)
                        if date_value:
                            break
                    
                    # If no attribute, try text content
                    if not date_value:
                        date_value = element.get_text(strip=True)
                    
                    if isinstance(date_value, list):
                        date_value = date_value[0] if date_value else None

                    if date_value:
                        parsed_date = self._parse_date_string(str(date_value))
                        if parsed_date:
                            # Higher confidence for time elements with datetime attribute
                            confidence = 0.85 if element.get('datetime') else 0.7
                            dates.append(DateInfo(
                                date=parsed_date,
                                confidence=confidence,
                                extraction_method=DateExtractionStrategy.TIME_ELEMENT,
                                source_text=date_value,
                                element_tag=element.name,
                                element_attributes=dict(element.attrs),
                            ))
        
        except Exception as e:
            logger.warning(f"Time element date extraction error: {e}")
        
        return dates
    
    async def _extract_from_url(self, url: str) -> List[DateInfo]:
        """Extract dates from URL patterns."""
        dates: List[DateInfo] = []
        
        try:
            parsed_url = urlparse(url)
            path = parsed_url.path
            
            for pattern in self.compiled_url_patterns:
                matches = pattern.finditer(path)
                
                for match in matches:
                    groups = match.groups()
                    
                    try:
                        if len(groups) == 3:
                            year, month, day = groups
                            # Handle different group formats
                            if len(year) == 4 and len(month) <= 2 and len(day) <= 2:
                                parsed_date = datetime(int(year), int(month), int(day))
                                dates.append(DateInfo(
                                    date=parsed_date,
                                    confidence=0.75,
                                    extraction_method=DateExtractionStrategy.URL_PATTERN,
                                    source_text=match.group(0),
                                ))
                    except (ValueError, TypeError):
                        continue
        
        except Exception as e:
            logger.warning(f"URL date extraction error: {e}")
        
        return dates
    
    async def _extract_from_content_patterns(self, html_content: str, text_content: str) -> List[DateInfo]:
        """Extract dates from content using regex patterns."""
        dates: List[DateInfo] = []
        
        try:
            # Combine HTML and text content
            content_to_search = f"{html_content}\n{text_content}"
            
            for pattern, format_type in self.compiled_patterns:
                matches = pattern.finditer(content_to_search)
                
                for match in matches:
                    date_string = match.group(1)
                    if date_string:
                        parsed_date = self._parse_date_string(date_string)
                        
                        if parsed_date:
                            # Lower confidence for content patterns
                            confidence = 0.6
                            
                            # Boost confidence if found in likely date contexts
                            context = content_to_search[max(0, match.start()-50):match.end()+50].lower()
                            if any(word in context for word in ['published', 'posted', 'created', 'updated']):
                                confidence += 0.1
                            
                            dates.append(DateInfo(
                                date=parsed_date,
                                confidence=min(0.8, confidence),
                                extraction_method=DateExtractionStrategy.CONTENT_PATTERN,
                                source_text=date_string,
                                date_format=format_type,
                            ))
        
        except Exception as e:
            logger.warning(f"Content pattern date extraction error: {e}")
        
        return dates
    
    def _parse_date_string(self, date_string: str, format_hint: Optional[str] = None) -> Optional[datetime]:
        """Parse a date string into a datetime object."""
        if not date_string:
            return None
        
        if not dateutil_parser:
            return self._manual_date_parse(date_string, format_hint)

        try:
            # Clean the date string
            date_string = str(date_string).strip()
            
            # Handle specific format hints
            if format_hint == 'unix_timestamp':
                timestamp = int(date_string)
                return datetime.fromtimestamp(timestamp)
            elif format_hint == 'unix_timestamp_ms':
                timestamp_ms = int(date_string) 
                timestamp = float(timestamp_ms) / 1000.0
                return datetime.fromtimestamp(timestamp)
            
            # Try dateutil parser if available (most flexible)
            if HAS_DATEUTIL and dateutil_parser:
                try:
                    return dateutil_parser.parse(date_string)
                except (ValueError, TypeError):
                    pass
            
            # Fallback to manual parsing for common formats
            return self._manual_date_parse(date_string, format_hint)
        
        except Exception as e:
            logger.debug(f"Date parsing failed for '{date_string}': {e}")
            return None
    
    def _manual_date_parse(self, date_string: str, format_hint: Optional[str] = None) -> Optional[datetime]:
        """Manual date parsing for common formats."""
        try:
            # ISO format
            if format_hint == 'iso_datetime' or 'T' in date_string:
                # Handle timezone info
                if date_string.endswith('Z'):
                    date_string = date_string[:-1]
                elif '+' in date_string or date_string.count('-') > 2:
                    # Remove timezone offset
                    date_string = re.sub(r'[+-]\d{2}:?\d{2}$', '', date_string)
                
                # Parse datetime
                if '.' in date_string:
                    return datetime.strptime(date_string.split('.')[0], '%Y-%m-%dT%H:%M:%S')
                else:
                    return datetime.strptime(date_string, '%Y-%m-%dT%H:%M:%S')
            
            elif format_hint == 'iso_date':
                return datetime.strptime(date_string, '%Y-%m-%d')
            
            # US format (MM/DD/YYYY)
            elif format_hint == 'us_date':
                return datetime.strptime(date_string, '%m/%d/%Y')
            
            # Try common formats
            common_formats = [
                '%Y-%m-%d',
                '%m/%d/%Y',
                '%d/%m/%Y',
                '%Y/%m/%d',
                '%m-%d-%Y',
                '%d-%m-%Y',
                '%m.%d.%Y',
                '%d.%m.%Y',
                '%B %d, %Y',
                '%b %d, %Y',
                '%d %B %Y',
                '%d %b %Y',
            ]
            
            for fmt in common_formats:
                try:
                    return datetime.strptime(date_string, fmt)
                except ValueError:
                    continue
        
        except Exception as e:
            logger.debug(f"Manual date parsing failed for '{date_string}': {e}")
        
        return None
    
    def _is_reasonable_date(self, date_obj: datetime) -> bool:
        """Check if a date is reasonable for a publication date."""
        # Make now() timezone-aware if date_obj is
        now = datetime.now(date_obj.tzinfo) if date_obj.tzinfo else datetime.now()
        current_year = now.year
        
        # Must be between 1990 and current year + 1
        if date_obj.year < 1990 or date_obj.year > current_year + 1:
            return False
        
        # Can't be in the future (with small tolerance)
        if date_obj > now.replace(hour=23, minute=59, second=59):
            return False
        
        return True
    
    async def extract_all_dates(
        self,
        html_content: str,
        url: str = "",
        text_content: str = "",
    ) -> List[DateInfo]:
        """
        Extract all possible dates from content.
        
        Returns all date candidates sorted by confidence.
        """
        candidates: List[DateInfo] = []
        
        # Extract from all strategies
        structured_dates = await self._extract_from_structured_data(html_content)
        candidates.extend(structured_dates)
        
        meta_dates = await self._extract_from_meta_tags(html_content)
        candidates.extend(meta_dates)
        
        if HAS_BS4:
            time_dates = await self._extract_from_time_elements(html_content)
            candidates.extend(time_dates)
        
        if url:
            url_dates = await self._extract_from_url(url)
            candidates.extend(url_dates)
        
        content_dates = await self._extract_from_content_patterns(html_content, text_content)
        candidates.extend(content_dates)
        
        # Filter reasonable dates and sort by confidence
        valid_dates = [d for d in candidates if self._is_reasonable_date(d.date)]
        valid_dates.sort(key=lambda x: x.confidence, reverse=True)
        
        return valid_dates 