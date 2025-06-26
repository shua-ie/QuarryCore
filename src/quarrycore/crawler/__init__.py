"""
QuarryCore Crawler Module - Adaptive Web Scraping Engine

This module provides a production-grade web crawler with HTTP/2 support,
adaptive concurrency, and intelligent rate limiting that scales from
Raspberry Pi (100-200 docs/min) to GPU workstations (1000-2000 docs/min).

Key Features:
- HTTP/2 persistent connections with smart pooling
- Adaptive semaphore-based concurrency (CPU_cores × 5)
- Circuit breaker pattern for failing domains
- Exponential backoff with jitter
- Robots.txt caching with TTL
- User-agent rotation and response streaming
- Bandwidth throttling and ETag support
- Playwright fallback for JavaScript-heavy sites
- Domain-specific rate limiting with automatic adjustment
"""

from .adaptive_crawler import AdaptiveCrawler
from .circuit_breaker import CircuitBreaker
from .rate_limiter import DomainRateLimiter
from .robots_parser import RobotsCache
from .user_agents import UserAgentRotator

__all__ = [
    "AdaptiveCrawler",
    "CircuitBreaker", 
    "DomainRateLimiter",
    "RobotsCache",
    "UserAgentRotator",
] 