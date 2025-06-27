"""
Social Metrics Extractor - Social Media Engagement Analysis

Extracts social media metrics including shares, likes, comments, and engagement
data from HTML content for comprehensive content analysis.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Pattern, Set

# BeautifulSoup imports with graceful fallbacks (proven pattern)
try:
    from bs4 import BeautifulSoup

    HAS_BS4 = True
except ImportError:
    if TYPE_CHECKING:
        from bs4 import BeautifulSoup
    else:
        BeautifulSoup = None  # type: ignore[misc,assignment]
    HAS_BS4 = False

logger = logging.getLogger(__name__)


@dataclass
class PlatformMetrics:
    """Social media metrics for a specific platform."""

    platform: str
    shares: Optional[int] = None
    likes: Optional[int] = None
    comments: Optional[int] = None
    reactions: Optional[int] = None

    # Platform-specific metrics
    retweets: Optional[int] = None  # Twitter
    favorites: Optional[int] = None  # Twitter
    pins: Optional[int] = None  # Pinterest
    upvotes: Optional[int] = None  # Reddit
    downvotes: Optional[int] = None  # Reddit

    # Engagement metadata
    url: Optional[str] = None
    profile_url: Optional[str] = None

    def total_engagement(self) -> int:
        """Calculate total engagement across all metrics."""
        total = 0
        for value in [
            self.shares,
            self.likes,
            self.comments,
            self.reactions,
            self.retweets,
            self.favorites,
            self.pins,
            self.upvotes,
        ]:
            if value is not None:
                total += value
        return total


@dataclass
class SocialMetrics:
    """Comprehensive social media metrics for content."""

    # Platform-specific metrics
    platforms: Dict[str, PlatformMetrics] = field(default_factory=dict)

    # Aggregated metrics
    total_shares: int = 0
    total_likes: int = 0
    total_comments: int = 0
    total_engagement: int = 0

    # Social proof indicators
    has_social_sharing: bool = False
    has_comment_system: bool = False
    has_like_system: bool = False

    # Additional metrics
    social_media_mentions: List[str] = field(default_factory=list)
    hashtags: List[str] = field(default_factory=list)

    # Extraction metadata
    extraction_method: str = "unknown"
    confidence_score: float = 0.0

    def __post_init__(self) -> None:
        """Calculate aggregated metrics."""
        self.total_shares = sum(p.shares or 0 for p in self.platforms.values())
        self.total_likes = sum(p.likes or 0 for p in self.platforms.values())
        self.total_comments = sum(p.comments or 0 for p in self.platforms.values())
        self.total_engagement = sum(p.total_engagement() for p in self.platforms.values())


class SocialMetricsExtractor:
    """
    Social media metrics extraction system.

    Extracts engagement metrics from social sharing widgets,
    embedded counters, and social media platform data.
    """

    def __init__(self) -> None:
        # Social platform patterns
        self.platform_patterns: Dict[str, Dict[str, List[str]]] = {
            "facebook": {
                "shares": [
                    r'facebook[^>]*share[^>]*["\'](\d+)["\']',
                    r'fb[_-]share[^>]*["\'](\d+)["\']',
                    r'data-share-count["\'](\d+)["\']',
                ],
                "likes": [
                    r'facebook[^>]*like[^>]*["\'](\d+)["\']',
                    r'fb[_-]like[^>]*["\'](\d+)["\']',
                ],
                "comments": [
                    r'facebook[^>]*comment[^>]*["\'](\d+)["\']',
                    r'fb[_-]comment[^>]*["\'](\d+)["\']',
                ],
            },
            "twitter": {
                "shares": [
                    r'twitter[^>]*share[^>]*["\'](\d+)["\']',
                    r'tweet[_-]count[^>]*["\'](\d+)["\']',
                ],
                "retweets": [
                    r'retweet[^>]*["\'](\d+)["\']',
                    r'rt[_-]count[^>]*["\'](\d+)["\']',
                ],
                "likes": [
                    r'twitter[^>]*like[^>]*["\'](\d+)["\']',
                    r'favorite[^>]*["\'](\d+)["\']',
                ],
            },
            "linkedin": {
                "shares": [
                    r'linkedin[^>]*share[^>]*["\'](\d+)["\']',
                    r'li[_-]share[^>]*["\'](\d+)["\']',
                ],
            },
            "pinterest": {
                "pins": [
                    r'pinterest[^>]*pin[^>]*["\'](\d+)["\']',
                    r'pin[_-]count[^>]*["\'](\d+)["\']',
                ],
            },
            "reddit": {
                "upvotes": [
                    r'reddit[^>]*up[^>]*["\'](\d+)["\']',
                    r'upvote[^>]*["\'](\d+)["\']',
                ],
                "downvotes": [
                    r'reddit[^>]*down[^>]*["\'](\d+)["\']',
                    r'downvote[^>]*["\'](\d+)["\']',
                ],
                "comments": [
                    r'reddit[^>]*comment[^>]*["\'](\d+)["\']',
                ],
            },
        }

        # Compile patterns
        self.compiled_patterns: Dict[str, Dict[str, List[Pattern[str]]]] = {}
        for platform, metrics in self.platform_patterns.items():
            self.compiled_patterns[platform] = {}
            for metric, patterns in metrics.items():
                self.compiled_patterns[platform][metric] = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]

        # Generic social sharing selectors
        self.social_selectors: List[str] = [
            # Share buttons and counters
            '[class*="share"]',
            '[class*="social"]',
            "[data-share-count]",
            "[data-share-url]",
            # Like buttons
            '[class*="like"]',
            '[class*="favorite"]',
            "[data-like-count]",
            # Comment counters
            '[class*="comment"]',
            "[data-comment-count]",
            # Platform-specific
            ".fb-share-button",
            ".twitter-share-button",
            ".linkedin-share-button",
            ".pinterest-share-button",
            # Generic counters
            ".social-count",
            ".share-count",
            ".engagement-count",
        ]

        # Hashtag pattern
        self.hashtag_pattern: Pattern[str] = re.compile(r"#(\w+)", re.IGNORECASE)

        # Social media mention patterns
        self.mention_patterns: Dict[str, Pattern[str]] = {
            "twitter": re.compile(r"@(\w+)", re.IGNORECASE),
            "instagram": re.compile(r"@(\w+)", re.IGNORECASE),
            "facebook": re.compile(r"facebook\.com/([^/\s]+)", re.IGNORECASE),
            "linkedin": re.compile(r"linkedin\.com/in/([^/\s]+)", re.IGNORECASE),
        }

        logger.info("SocialMetricsExtractor initialized")

    async def extract_metrics(self, html_content: str) -> Optional[SocialMetrics]:
        """
        Extract social media metrics from HTML content.

        Args:
            html_content: HTML content to analyze

        Returns:
            SocialMetrics object with extracted data, or None if no metrics found
        """
        metrics = SocialMetrics()

        try:
            # Extract platform-specific metrics
            await self._extract_platform_metrics(html_content, metrics)

            # Extract social sharing indicators
            await self._extract_social_indicators(html_content, metrics)

            # Extract hashtags and mentions
            await self._extract_hashtags_mentions(html_content, metrics)

            # Extract from structured data
            await self._extract_from_structured_data(html_content, metrics)

            # Calculate confidence score
            metrics.confidence_score = self._calculate_confidence(metrics)

            # Only return if we found meaningful data
            if metrics.total_engagement > 0 or metrics.has_social_sharing:
                return metrics

        except Exception as e:
            logger.error(f"Social metrics extraction error: {e}")

        return None

    async def _extract_platform_metrics(self, html_content: str, metrics: SocialMetrics) -> None:
        """Extract metrics using platform-specific patterns."""
        for platform, metric_patterns in self.compiled_patterns.items():
            platform_metrics = PlatformMetrics(platform=platform)
            found_metrics = False

            for metric_name, patterns in metric_patterns.items():
                for pattern in patterns:
                    matches = pattern.finditer(html_content)

                    for match in matches:
                        try:
                            count = int(match.group(1))
                            setattr(platform_metrics, metric_name, count)
                            found_metrics = True
                            break
                        except (ValueError, IndexError):
                            continue

                    if getattr(platform_metrics, metric_name) is not None:
                        break

            if found_metrics:
                metrics.platforms[platform] = platform_metrics

    async def _extract_social_indicators(self, html_content: str, metrics: SocialMetrics) -> None:
        """Extract social sharing and engagement indicators."""
        if not HAS_BS4:
            await self._extract_indicators_regex(html_content, metrics)
            return

        try:
            soup = BeautifulSoup(html_content, "html.parser")

            # Check for social sharing elements
            for selector in self.social_selectors:
                elements = soup.select(selector)

                if elements:
                    # Determine what type of social element this is
                    for element in elements:
                        element_text = element.get_text(strip=True).lower()
                        element_class = " ".join(element.get("class", [])).lower()

                        # Check for sharing functionality
                        if any(word in element_class or word in element_text for word in ["share", "social"]):
                            metrics.has_social_sharing = True

                        # Check for like functionality
                        if any(word in element_class or word in element_text for word in ["like", "favorite", "heart"]):
                            metrics.has_like_system = True

                        # Check for comment functionality
                        if any(
                            word in element_class or word in element_text for word in ["comment", "reply", "discuss"]
                        ):
                            metrics.has_comment_system = True

                        # Try to extract numeric values
                        numbers = re.findall(r"\b(\d+)\b", element_text)
                        if numbers:
                            try:
                                count = int(numbers[0])
                                if count > 0:
                                    # Try to determine what this count represents
                                    if "share" in element_class or "share" in element_text:
                                        metrics.total_shares += count
                                    elif "like" in element_class or "like" in element_text:
                                        metrics.total_likes += count
                                    elif "comment" in element_class or "comment" in element_text:
                                        metrics.total_comments += count
                            except ValueError:
                                continue

            # Look for embedded social media widgets
            social_widgets = [
                'iframe[src*="facebook.com"]',
                'iframe[src*="twitter.com"]',
                'iframe[src*="instagram.com"]',
                'iframe[src*="youtube.com"]',
                'div[class*="fb-"]',
                'div[class*="twitter-"]',
                'div[class*="instagram-"]',
            ]

            for widget_selector in social_widgets:
                if soup.select(widget_selector):
                    metrics.has_social_sharing = True
                    break

        except Exception as e:
            logger.warning(f"Social indicators extraction error: {e}")

    async def _extract_indicators_regex(self, html_content: str, metrics: SocialMetrics) -> None:
        """Fallback regex-based indicator extraction."""
        # Check for social sharing patterns
        share_patterns = [
            r'class=["\'][^"\']*share[^"\']*["\']',
            r"share[_-]button",
            r"social[_-]share",
            r"facebook\.com/sharer",
            r"twitter\.com/intent/tweet",
        ]

        for pattern in share_patterns:
            if re.search(pattern, html_content, re.IGNORECASE):
                metrics.has_social_sharing = True
                break

        # Check for like systems
        like_patterns = [
            r'class=["\'][^"\']*like[^"\']*["\']',
            r"like[_-]button",
            r"favorite[_-]button",
            r"heart[_-]button",
        ]

        for pattern in like_patterns:
            if re.search(pattern, html_content, re.IGNORECASE):
                metrics.has_like_system = True
                break

        # Check for comment systems
        comment_patterns = [
            r'class=["\'][^"\']*comment[^"\']*["\']',
            r"comment[_-]system",
            r"disqus",
            r"livefyre",
            r"facebook\.com/plugins/comments",
        ]

        for pattern in comment_patterns:
            if re.search(pattern, html_content, re.IGNORECASE):
                metrics.has_comment_system = True
                break

    async def _extract_hashtags_mentions(self, html_content: str, metrics: SocialMetrics) -> None:
        """Extract hashtags and social media mentions."""
        try:
            # Extract hashtags
            hashtag_matches = self.hashtag_pattern.finditer(html_content)
            hashtags: Set[str] = set()

            for match in hashtag_matches:
                hashtag = match.group(1).lower()
                # Filter out common false positives
                if len(hashtag) > 1 and not hashtag.isdigit():
                    hashtags.add(hashtag)

            metrics.hashtags = list(hashtags)[:20]  # Limit to 20 hashtags

            # Extract social media mentions
            mentions: Set[str] = set()

            for _platform, pattern in self.mention_patterns.items():
                platform_matches = pattern.finditer(html_content)

                for match in platform_matches:
                    mention = match.group(0)
                    mentions.add(mention)

            metrics.social_media_mentions = list(mentions)[:10]  # Limit to 10 mentions

        except Exception as e:
            logger.warning(f"Hashtag/mention extraction error: {e}")

    async def _extract_from_structured_data(self, html_content: str, metrics: SocialMetrics) -> None:
        """Extract metrics from structured data (JSON-LD)."""
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
                        # Look for interaction statistics
                        interaction_stats = item.get("interactionStatistic", [])
                        if not isinstance(interaction_stats, list):
                            interaction_stats = [interaction_stats]

                        for stat in interaction_stats:
                            if isinstance(stat, dict):
                                interaction_type = stat.get("interactionType", {})
                                user_interaction_count = stat.get("userInteractionCount", 0)

                                if isinstance(interaction_type, dict):
                                    type_name = interaction_type.get("@type", "").lower()

                                    if "like" in type_name:
                                        metrics.total_likes += int(user_interaction_count)
                                    elif "share" in type_name:
                                        metrics.total_shares += int(user_interaction_count)
                                    elif "comment" in type_name:
                                        metrics.total_comments += int(user_interaction_count)

                        # Look for social media accounts
                        same_as = item.get("sameAs", [])
                        if isinstance(same_as, list):
                            for url in same_as:
                                if any(
                                    platform in url.lower()
                                    for platform in [
                                        "facebook",
                                        "twitter",
                                        "instagram",
                                        "linkedin",
                                    ]
                                ):
                                    metrics.social_media_mentions.append(url)

                except json.JSONDecodeError:
                    continue

        except Exception as e:
            logger.warning(f"Structured data social metrics extraction error: {e}")

    def _calculate_confidence(self, metrics: SocialMetrics) -> float:
        """Calculate confidence score for extracted metrics."""
        confidence = 0.0

        # Base confidence from platform metrics
        if metrics.platforms:
            confidence += 0.4

            # Bonus for multiple platforms
            if len(metrics.platforms) > 1:
                confidence += 0.2

        # Confidence from engagement numbers
        if metrics.total_engagement > 0:
            confidence += 0.3

            # Bonus for high engagement
            if metrics.total_engagement > 100:
                confidence += 0.1

        # Confidence from social indicators
        indicator_count = sum(
            [
                metrics.has_social_sharing,
                metrics.has_like_system,
                metrics.has_comment_system,
            ]
        )
        confidence += indicator_count * 0.1

        # Confidence from hashtags and mentions
        if metrics.hashtags:
            confidence += 0.05
        if metrics.social_media_mentions:
            confidence += 0.05

        return min(1.0, confidence)

    async def extract_detailed_metrics(self, html_content: str) -> Dict[str, Any]:
        """
        Extract detailed social metrics with analysis.

        Returns comprehensive social media analysis including
        engagement levels, platform distribution, and social proof assessment.
        """
        metrics = await self.extract_metrics(html_content)

        if not metrics:
            return {
                "metrics": None,
                "analysis": {
                    "engagement_level": "none",
                    "primary_platforms": [],
                    "social_proof": "none",
                    "recommendations": [
                        "Add social sharing buttons",
                        "Enable social login",
                        "Add social media widgets",
                    ],
                },
            }

        # Detailed analysis
        result: Dict[str, Any] = {
            "metrics": {
                "platforms": {
                    name: {
                        "shares": platform.shares,
                        "likes": platform.likes,
                        "comments": platform.comments,
                        "total_engagement": platform.total_engagement(),
                    }
                    for name, platform in metrics.platforms.items()
                },
                "totals": {
                    "shares": metrics.total_shares,
                    "likes": metrics.total_likes,
                    "comments": metrics.total_comments,
                    "engagement": metrics.total_engagement,
                },
                "indicators": {
                    "has_social_sharing": metrics.has_social_sharing,
                    "has_like_system": metrics.has_like_system,
                    "has_comment_system": metrics.has_comment_system,
                },
                "content": {
                    "hashtags": metrics.hashtags,
                    "mentions": metrics.social_media_mentions,
                },
                "confidence": metrics.confidence_score,
            },
            "analysis": {
                "engagement_level": self._classify_engagement_level(metrics),
                "primary_platforms": self._identify_primary_platforms(metrics),
                "social_proof": self._assess_social_proof(metrics),
            },
        }

        # Add recommendations based on analysis
        recommendations = []
        if not metrics.has_social_sharing:
            recommendations.append("Add social sharing buttons")
        if not metrics.has_like_system:
            recommendations.append("Consider adding like/favorite functionality")
        if not metrics.has_comment_system:
            recommendations.append("Enable comment system for user engagement")
        if metrics.total_engagement < 10:
            recommendations.append("Focus on content promotion to increase engagement")

        result["analysis"]["recommendations"] = recommendations

        return result

    def _classify_engagement_level(self, metrics: SocialMetrics) -> str:
        """Classify the overall engagement level."""
        total = metrics.total_engagement

        if total == 0:
            return "none"
        elif total < 10:
            return "low"
        elif total < 100:
            return "moderate"
        elif total < 1000:
            return "high"
        else:
            return "viral"

    def _identify_primary_platforms(self, metrics: SocialMetrics) -> List[str]:
        """Identify the primary social media platforms."""
        platform_scores = []

        for name, platform in metrics.platforms.items():
            score = platform.total_engagement()
            if score > 0:
                platform_scores.append((name, score))

        # Sort by engagement and return top platforms
        platform_scores.sort(key=lambda x: x[1], reverse=True)
        return [name for name, _ in platform_scores[:3]]

    def _assess_social_proof(self, metrics: SocialMetrics) -> str:
        """Assess the level of social proof."""
        indicators = [
            metrics.has_social_sharing,
            metrics.has_like_system,
            metrics.has_comment_system,
            len(metrics.platforms) > 0,
            metrics.total_engagement > 0,
            len(metrics.social_media_mentions) > 0,
        ]

        score = sum(indicators) / len(indicators)

        if score >= 0.8:
            return "strong"
        elif score >= 0.5:
            return "moderate"
        elif score >= 0.2:
            return "weak"
        else:
            return "none"
