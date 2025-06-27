"""
Structured Data Parser - OpenGraph, Schema.org, and Twitter Cards

Extracts structured metadata from HTML content using multiple parsing strategies
for comprehensive content analysis in AI training data mining.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from urllib.parse import urljoin

# BeautifulSoup imports with graceful fallbacks
try:
    from bs4 import BeautifulSoup, NavigableString, Tag

    HAS_BS4 = True
except ImportError:
    if TYPE_CHECKING:
        from bs4 import BeautifulSoup, NavigableString, Tag
    else:
        BeautifulSoup = None  # type: ignore[misc,assignment]
        Tag = None  # type: ignore[misc,assignment]
        NavigableString = None  # type: ignore[misc,assignment]
    HAS_BS4 = False

logger = logging.getLogger(__name__)


@dataclass
class StructuredDataResult:
    """Result of structured data extraction."""

    # OpenGraph data
    og_title: Optional[str] = None
    og_description: Optional[str] = None
    og_image: Optional[str] = None
    og_url: Optional[str] = None
    og_type: Optional[str] = None
    og_site_name: Optional[str] = None
    og_locale: Optional[str] = None
    og_updated_time: Optional[str] = None

    # Schema.org data
    schema_type: Optional[str] = None
    schema_title: Optional[str] = None
    schema_description: Optional[str] = None
    schema_author: Optional[str] = None
    schema_date_published: Optional[str] = None
    schema_date_modified: Optional[str] = None
    schema_image: Optional[str] = None
    schema_publisher: Optional[str] = None

    # Twitter Cards
    twitter_card: Optional[str] = None
    twitter_title: Optional[str] = None
    twitter_description: Optional[str] = None
    twitter_image: Optional[str] = None
    twitter_site: Optional[str] = None
    twitter_creator: Optional[str] = None

    # Additional metadata
    canonical_url: Optional[str] = None
    meta_title: Optional[str] = None
    meta_description: Optional[str] = None
    meta_keywords: Optional[str] = None
    meta_author: Optional[str] = None
    meta_robots: Optional[str] = None

    # Raw structured data
    raw_json_ld: List[Dict[str, Any]] = field(default_factory=list)
    raw_microdata: Dict[str, Any] = field(default_factory=dict)

    # Extraction metadata
    extraction_method: str = "unknown"
    confidence_score: float = 0.0
    errors: List[str] = field(default_factory=list)


class OpenGraphParser:
    """Parser for OpenGraph metadata."""

    @staticmethod
    def parse(soup: Any, base_url: str = "") -> Dict[str, Any]:
        """Parse OpenGraph metadata from HTML soup."""
        og_data: Dict[str, Any] = {}

        try:
            # Find all OpenGraph meta tags
            og_tags = soup.find_all("meta", property=re.compile(r"^og:"))

            for tag in og_tags:
                property_name = tag.get("property", "")
                content = tag.get("content", "")

                if property_name and content:
                    # Clean property name (remove og: prefix)
                    clean_name = property_name.replace("og:", "").replace(":", "_")

                    # Handle URLs
                    if clean_name in ["image", "url"] and content and base_url:
                        content = urljoin(base_url, content)

                    og_data[f"og_{clean_name}"] = content

            # Also check for Facebook-specific tags
            fb_tags = soup.find_all("meta", property=re.compile(r"^fb:"))
            for tag in fb_tags:
                property_name = tag.get("property", "")
                content = tag.get("content", "")

                if property_name and content:
                    clean_name = property_name.replace("fb:", "").replace(":", "_")
                    og_data[f"fb_{clean_name}"] = content

        except Exception as e:
            logger.warning(f"OpenGraph parsing error: {e}")

        return og_data


class SchemaOrgParser:
    """Parser for Schema.org structured data."""

    @staticmethod
    def parse_json_ld(soup: Any) -> List[Dict[str, Any]]:
        """Parse JSON-LD structured data."""
        json_ld_data: List[Dict[str, Any]] = []

        try:
            # Find all script tags with type application/ld+json
            scripts = soup.find_all("script", type="application/ld+json")

            for script in scripts:
                try:
                    if hasattr(script, "string") and script.string:
                        # Clean and parse JSON
                        json_text = str(script.string).strip()
                        data = json.loads(json_text)

                        # Handle both single objects and arrays
                        if isinstance(data, list):
                            json_ld_data.extend(data)
                        else:
                            json_ld_data.append(data)

                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON-LD: {e}")
                    continue

        except Exception as e:
            logger.warning(f"JSON-LD parsing error: {e}")

        return json_ld_data

    @staticmethod
    def parse_microdata(soup: Any) -> Dict[str, Any]:
        """Parse Microdata structured data."""
        microdata: Dict[str, Any] = {}

        try:
            # Find elements with itemscope
            items = soup.find_all(attrs={"itemscope": True})

            for item in items:
                item_type = item.get("itemtype", "")
                if not item_type:
                    continue

                # Extract properties
                properties: Dict[str, Any] = {}
                prop_elements = item.find_all(attrs={"itemprop": True})

                for prop_elem in prop_elements:
                    prop_name = prop_elem.get("itemprop")

                    # Get property value
                    if hasattr(prop_elem, "name") and prop_elem.name in ["meta"]:
                        prop_value = prop_elem.get("content", "")
                    elif hasattr(prop_elem, "name") and prop_elem.name in ["time"]:
                        prop_value = prop_elem.get("datetime", prop_elem.get_text().strip())
                    elif hasattr(prop_elem, "name") and prop_elem.name in ["img"]:
                        prop_value = prop_elem.get("src", "")
                    elif hasattr(prop_elem, "name") and prop_elem.name in ["a"]:
                        prop_value = prop_elem.get("href", prop_elem.get_text().strip())
                    else:
                        prop_value = prop_elem.get_text().strip()

                    if prop_name and prop_value:
                        properties[prop_name] = prop_value

                if properties:
                    microdata[item_type] = properties

        except Exception as e:
            logger.warning(f"Microdata parsing error: {e}")

        return microdata

    @staticmethod
    def extract_schema_fields(json_ld_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract common Schema.org fields from JSON-LD data."""
        schema_fields: Dict[str, Any] = {}

        for item in json_ld_data:
            try:
                # Get schema type
                schema_type = item.get("@type", "")
                if schema_type:
                    schema_fields["schema_type"] = schema_type

                # Extract common fields
                field_mappings = {
                    "name": "schema_title",
                    "headline": "schema_title",
                    "title": "schema_title",
                    "description": "schema_description",
                    "author": "schema_author",
                    "datePublished": "schema_date_published",
                    "dateModified": "schema_date_modified",
                    "image": "schema_image",
                    "publisher": "schema_publisher",
                }

                for json_key, schema_key in field_mappings.items():
                    value = item.get(json_key)
                    if value:
                        # Handle nested objects
                        if isinstance(value, dict):
                            if "name" in value:
                                schema_fields[schema_key] = value["name"]
                            elif "@id" in value:
                                schema_fields[schema_key] = value["@id"]
                        elif isinstance(value, list) and value:
                            # Take first item if it's a list
                            first_item = value[0]
                            if isinstance(first_item, dict) and "name" in first_item:
                                schema_fields[schema_key] = first_item["name"]
                            else:
                                schema_fields[schema_key] = str(first_item)
                        else:
                            schema_fields[schema_key] = str(value)

            except Exception as e:
                logger.warning(f"Schema field extraction error: {e}")
                continue

        return schema_fields


class TwitterCardParser:
    """Parser for Twitter Card metadata."""

    @staticmethod
    def parse(soup: Any) -> Dict[str, Any]:
        """Parse Twitter Card metadata from HTML soup."""
        twitter_data: Dict[str, Any] = {}

        try:
            # Find all Twitter meta tags
            twitter_tags = soup.find_all("meta", name=re.compile(r"^twitter:"))

            for tag in twitter_tags:
                name = tag.get("name", "")
                content = tag.get("content", "")

                if name and content:
                    # Clean name (remove twitter: prefix)
                    clean_name = name.replace("twitter:", "").replace(":", "_")
                    twitter_data[f"twitter_{clean_name}"] = content

        except Exception as e:
            logger.warning(f"Twitter Card parsing error: {e}")

        return twitter_data


class StructuredDataParser:
    """
    Comprehensive structured data parser.

    Extracts metadata from OpenGraph, Schema.org, Twitter Cards,
    and standard HTML meta tags for content analysis.
    """

    def __init__(self) -> None:
        self.og_parser = OpenGraphParser()
        self.schema_parser = SchemaOrgParser()
        self.twitter_parser = TwitterCardParser()

    async def parse_all(self, html_content: str, base_url: str = "") -> Dict[str, Any]:
        """
        Parse all structured data from HTML content.

        Args:
            html_content: HTML content to parse
            base_url: Base URL for resolving relative URLs

        Returns:
            Dictionary containing all extracted structured data
        """
        if not HAS_BS4:
            logger.warning("BeautifulSoup not available, using fallback parser")
            return await self._parse_fallback(html_content)

        try:
            # Parse HTML
            soup = BeautifulSoup(html_content, "html.parser")

            # Extract all structured data types
            result: Dict[str, Any] = {}

            # OpenGraph data
            og_data = self.og_parser.parse(soup, base_url)
            result.update(og_data)

            # Schema.org JSON-LD
            json_ld_data = self.schema_parser.parse_json_ld(soup)
            schema_fields = self.schema_parser.extract_schema_fields(json_ld_data)
            result.update(schema_fields)
            result["raw_json_ld"] = json_ld_data

            # Schema.org Microdata
            microdata = self.schema_parser.parse_microdata(soup)
            result["raw_microdata"] = microdata

            # Twitter Cards
            twitter_data = self.twitter_parser.parse(soup)
            result.update(twitter_data)

            # Standard HTML meta tags
            meta_data = await self._parse_standard_meta(soup)
            result.update(meta_data)

            # Canonical URL
            canonical = soup.find("link", rel="canonical")
            if canonical and hasattr(canonical, "get"):
                href = canonical.get("href")
                if href:
                    canonical_url = str(href)
                    if base_url:
                        canonical_url = urljoin(base_url, canonical_url)
                    result["canonical_url"] = canonical_url

            return result

        except Exception as e:
            logger.error(f"Structured data parsing failed: {e}")
            return await self._parse_fallback(html_content)

    async def _parse_standard_meta(self, soup: Any) -> Dict[str, Any]:
        """Parse standard HTML meta tags."""
        meta_data: Dict[str, Any] = {}

        try:
            # Title tag
            title_tag = soup.find("title")
            if title_tag and hasattr(title_tag, "string") and title_tag.string:
                meta_data["meta_title"] = str(title_tag.string).strip()

            # Meta description
            meta_desc = soup.find("meta", attrs={"name": "description"})
            if meta_desc and meta_desc.get("content"):
                meta_data["meta_description"] = str(meta_desc.get("content", ""))

            # Meta keywords
            meta_keywords = soup.find("meta", attrs={"name": "keywords"})
            if meta_keywords and meta_keywords.get("content"):
                meta_data["meta_keywords"] = str(meta_keywords.get("content", ""))

            # Meta author
            meta_author = soup.find("meta", attrs={"name": "author"})
            if meta_author and meta_author.get("content"):
                meta_data["meta_author"] = str(meta_author.get("content", ""))

            # Meta robots
            meta_robots = soup.find("meta", attrs={"name": "robots"})
            if meta_robots and meta_robots.get("content"):
                meta_data["meta_robots"] = str(meta_robots.get("content", ""))

            # Additional meta tags
            meta_tags = [
                ("generator", "meta_generator"),
                ("viewport", "meta_viewport"),
                ("theme-color", "meta_theme_color"),
                ("application-name", "meta_app_name"),
            ]

            for name, key in meta_tags:
                meta_tag = soup.find("meta", attrs={"name": name})
                if meta_tag and meta_tag.get("content"):
                    meta_data[key] = str(meta_tag.get("content", ""))

        except Exception as e:
            logger.warning(f"Standard meta parsing error: {e}")

        return meta_data

    async def _parse_fallback(self, html_content: str) -> Dict[str, Any]:
        """Fallback parser using regex when BeautifulSoup is not available."""
        result: Dict[str, Any] = {}

        try:
            # Extract title
            title_match = re.search(r"<title[^>]*>([^<]+)</title>", html_content, re.IGNORECASE)
            if title_match:
                result["meta_title"] = title_match.group(1).strip()

            # Extract meta tags using regex
            meta_pattern = r"<meta\s+([^>]+)>"
            meta_matches = re.finditer(meta_pattern, html_content, re.IGNORECASE)

            for match in meta_matches:
                meta_attrs = match.group(1)

                # Parse attributes
                attr_pattern = r'(\w+)=["\'](.*?)["\']'
                attrs = dict(re.findall(attr_pattern, meta_attrs))

                # OpenGraph
                if "property" in attrs and attrs["property"].startswith("og:"):
                    prop_name = attrs["property"].replace("og:", "").replace(":", "_")
                    if "content" in attrs:
                        result[f"og_{prop_name}"] = attrs["content"]

                # Twitter Cards
                elif "name" in attrs and attrs["name"].startswith("twitter:"):
                    name = attrs["name"].replace("twitter:", "").replace(":", "_")
                    if "content" in attrs:
                        result[f"twitter_{name}"] = attrs["content"]

                # Standard meta tags
                elif "name" in attrs and "content" in attrs:
                    name = attrs["name"].lower()
                    if name in ["description", "keywords", "author", "robots"]:
                        result[f"meta_{name}"] = attrs["content"]

            # Extract JSON-LD (basic)
            json_ld_pattern = r'<script[^>]*type=["\']application/ld\+json["\'][^>]*>(.*?)</script>'
            json_ld_matches = re.finditer(json_ld_pattern, html_content, re.IGNORECASE | re.DOTALL)

            json_ld_data: List[Dict[str, Any]] = []
            for match in json_ld_matches:
                try:
                    json_text = match.group(1).strip()
                    data = json.loads(json_text)
                    json_ld_data.append(data)
                except json.JSONDecodeError:
                    continue

            if json_ld_data:
                result["raw_json_ld"] = json_ld_data
                # Extract basic schema fields
                schema_fields = self.schema_parser.extract_schema_fields(json_ld_data)
                result.update(schema_fields)

        except Exception as e:
            logger.warning(f"Fallback parsing error: {e}")

        return result

    def calculate_confidence(self, structured_data: Dict[str, Any]) -> float:
        """Calculate confidence score for extracted structured data."""
        score = 0.0
        total_possible = 0.0

        # Weight different data sources
        weights = {
            "og_": 0.3,  # OpenGraph
            "schema_": 0.4,  # Schema.org
            "twitter_": 0.2,  # Twitter Cards
            "meta_": 0.1,  # Standard meta
        }

        for prefix, weight in weights.items():
            found_fields = sum(1 for key in structured_data.keys() if key.startswith(prefix))
            expected_fields = 5  # Reasonable number of expected fields per type

            if found_fields > 0:
                field_score = min(found_fields / expected_fields, 1.0)
                score += field_score * weight

            total_possible += weight

        # Bonus for having JSON-LD data
        if structured_data.get("raw_json_ld"):
            score += 0.1
            total_possible += 0.1

        # Normalize score
        if total_possible > 0:
            return min(score / total_possible, 1.0)

        return 0.0

    async def extract_structured_result(self, html_content: str, base_url: str = "") -> StructuredDataResult:
        """
        Extract structured data and return as StructuredDataResult object.

        Args:
            html_content: HTML content to parse
            base_url: Base URL for resolving relative URLs

        Returns:
            StructuredDataResult object with all extracted data
        """
        try:
            # Parse all structured data
            data = await self.parse_all(html_content, base_url)

            # Create result object
            result = StructuredDataResult(
                # OpenGraph fields
                og_title=data.get("og_title"),
                og_description=data.get("og_description"),
                og_image=data.get("og_image"),
                og_url=data.get("og_url"),
                og_type=data.get("og_type"),
                og_site_name=data.get("og_site_name"),
                og_locale=data.get("og_locale"),
                og_updated_time=data.get("og_updated_time"),
                # Schema.org fields
                schema_type=data.get("schema_type"),
                schema_title=data.get("schema_title"),
                schema_description=data.get("schema_description"),
                schema_author=data.get("schema_author"),
                schema_date_published=data.get("schema_date_published"),
                schema_date_modified=data.get("schema_date_modified"),
                schema_image=data.get("schema_image"),
                schema_publisher=data.get("schema_publisher"),
                # Twitter Card fields
                twitter_card=data.get("twitter_card"),
                twitter_title=data.get("twitter_title"),
                twitter_description=data.get("twitter_description"),
                twitter_image=data.get("twitter_image"),
                twitter_site=data.get("twitter_site"),
                twitter_creator=data.get("twitter_creator"),
                # Meta fields
                canonical_url=data.get("canonical_url"),
                meta_title=data.get("meta_title"),
                meta_description=data.get("meta_description"),
                meta_keywords=data.get("meta_keywords"),
                meta_author=data.get("meta_author"),
                meta_robots=data.get("meta_robots"),
                # Raw data
                raw_json_ld=data.get("raw_json_ld", []),
                raw_microdata=data.get("raw_microdata", {}),
                # Metadata
                extraction_method="beautifulsoup" if HAS_BS4 else "regex_fallback",
                confidence_score=self.calculate_confidence(data),
            )

            return result

        except Exception as e:
            logger.error(f"Structured data extraction failed: {e}")
            return StructuredDataResult(
                extraction_method="error",
                confidence_score=0.0,
                errors=[str(e)],
            )
