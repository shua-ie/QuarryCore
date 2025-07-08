"""
Production-quality HTTP client with robots.txt compliance, rate limiting, and observability.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlparse

import aiohttp
import async_timeout
import robotstxt
import structlog
from selectolax.parser import HTMLParser

from quarrycore.config.config import Config
from quarrycore.observability.metrics import METRICS

logger = structlog.get_logger(__name__)


class RobotsTxtCache:
    """Simple robots.txt cache with async interface."""

    def __init__(self, session: aiohttp.ClientSession, cache_ttl: int = 12 * 60 * 60):
        self.session = session
        self.cache_ttl = cache_ttl
        self._cache: Dict[str, tuple[float, Optional[robotstxt.RobotsFile]]] = {}
        self._lock = asyncio.Lock()

    async def can_fetch(self, url: str, user_agent: str) -> bool:
        """Check if user agent can fetch the given URL."""
        try:
            parsed_url = urlparse(url)
            domain = parsed_url.netloc
            robots_url = f"{parsed_url.scheme}://{domain}/robots.txt"

            # Check cache
            async with self._lock:
                cached_entry = self._cache.get(domain)
                if cached_entry:
                    cached_time, robots_file = cached_entry
                    if time.time() - cached_time < self.cache_ttl:
                        if robots_file is None:
                            return True  # No robots.txt found, allow
                        result = robots_file.test_url(url, user_agent)
                        return not result.get("disallowed", False)

            # Fetch robots.txt
            try:
                async with self.session.get(robots_url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status == 200:
                        content = await response.text()
                        robots_file = robotstxt.RobotsFile(content)
                    else:
                        robots_file = None  # No robots.txt or error
            except Exception:
                robots_file = None  # Allow by default on fetch error

            # Cache result
            async with self._lock:
                self._cache[domain] = (time.time(), robots_file)

            # Check permission
            if robots_file is None:
                return True

            result = robots_file.test_url(url, user_agent)
            return not result.get("disallowed", False)

        except Exception as e:
            logger.debug("Error checking robots.txt", url=url, error=str(e))
            return True  # Allow by default on error


@dataclass
class CrawlerResponse:
    """Response from HTTP crawling with timing and attempt information."""

    status: int
    headers: Dict[str, str]
    body: bytes
    start_ts: float
    end_ts: float
    attempts: int
    url: str
    final_url: str


class DomainBackoffRegistry:
    """Registry for tracking domain-level backoff and cooldown periods."""

    def __init__(self, cooldown_seconds: float = 120):
        self.cooldown_seconds = cooldown_seconds
        self._failures: Dict[str, int] = {}
        self._cooldown_until: Dict[str, float] = {}
        self._lock = asyncio.Lock()

    async def record_failure(self, domain: str) -> None:
        """Record a failure for a domain."""
        async with self._lock:
            self._failures[domain] = self._failures.get(domain, 0) + 1
            if self._failures[domain] >= 3:
                self._cooldown_until[domain] = time.time() + self.cooldown_seconds
                logger.warning(
                    "Domain entered cooldown period",
                    domain=domain,
                    failures=self._failures[domain],
                    cooldown_until=self._cooldown_until[domain],
                )

    async def record_success(self, domain: str) -> None:
        """Record a success for a domain, clearing failure count."""
        async with self._lock:
            self._failures.pop(domain, None)
            self._cooldown_until.pop(domain, None)

    async def is_in_cooldown(self, domain: str) -> bool:
        """Check if domain is in cooldown period."""
        async with self._lock:
            cooldown_time = self._cooldown_until.get(domain, 0)
            if cooldown_time > time.time():
                return True
            # Clean up expired cooldown
            if domain in self._cooldown_until:
                del self._cooldown_until[domain]
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get current backoff statistics."""
        current_time = time.time()
        return {
            "domains_in_cooldown": sum(
                1 for cooldown_time in self._cooldown_until.values() if cooldown_time > current_time
            ),
            "total_failures": dict(self._failures),
            "cooldown_domains": {
                domain: cooldown_time - current_time
                for domain, cooldown_time in self._cooldown_until.items()
                if cooldown_time > current_time
            },
        }

    def cleanup(self) -> None:
        """Remove expired cooldown entries to free memory."""
        current_time = time.time()
        expired_domains = [
            domain for domain, cooldown_time in self._cooldown_until.items() if cooldown_time <= current_time
        ]

        for domain in expired_domains:
            self._cooldown_until.pop(domain, None)
            # Note: We keep failure counts as they represent historical data
            # Only remove from cooldown tracking

        if expired_domains:
            logger.debug(
                "Cleaned up expired domain cooldowns", expired_count=len(expired_domains), domains=expired_domains
            )


class HttpClient:
    """Production-quality HTTP client with robots.txt compliance and rate limiting."""

    _last_loop_id: Optional[int] = None

    def __init__(self, config: Config):
        self.config = config
        self.crawler_config = config.crawler

        # Global TCP connector with connection pooling
        self.connector = aiohttp.TCPConnector(
            limit=0,  # No global limit
            ttl_dns_cache=30,
            use_dns_cache=True,
            keepalive_timeout=30,
            enable_cleanup_closed=True,
        )

        # Per-domain semaphores for concurrency control
        self._domain_semaphores: Dict[str, asyncio.Semaphore] = {}
        self._semaphore_lock = asyncio.Lock()

        # Domain backoff registry
        self.backoff_registry = DomainBackoffRegistry(cooldown_seconds=self.crawler_config.backoff_cooldown_seconds)

        # Proxy rotation setup
        self.proxies: List[str] = []
        self._proxy_index = 0
        self._setup_proxies()

        # Session will be initialized in initialize()
        self.session: Optional[aiohttp.ClientSession] = None
        self._is_initialized = False

        # Metrics tracking
        self._in_flight_requests = 0
        self._domain_inflight: Dict[str, int] = {}  # Track concurrent requests per domain

        logger.info(
            "HTTP client initialized",
            max_concurrency_per_domain=self.crawler_config.max_concurrency_per_domain,
            max_retries=self.crawler_config.max_retries,
            user_agent=self.crawler_config.user_agent,
            proxies_count=len(self.proxies),
        )

    def _setup_proxies(self) -> None:
        """Setup proxy rotation from environment variable."""
        proxy_env = os.environ.get("QUARRY_HTTP_PROXIES", "")
        if proxy_env:
            self.proxies = [proxy.strip() for proxy in proxy_env.split(",") if proxy.strip()]
            # Use deterministic ordering in test mode
            if not self.config.debug.test_mode:
                random.shuffle(self.proxies)  # Randomize initial order
            logger.info("Proxy rotation enabled", proxy_count=len(self.proxies))

    def _get_next_proxy(self, url: str) -> Optional[str]:
        """Get next proxy for URL with scheme matching."""
        if not self.proxies:
            return None

        # Use deterministic selection in test mode
        if self.config.debug.test_mode:
            return self._select_proxy_deterministic(url)

        parsed_url = urlparse(url)
        url_scheme = parsed_url.scheme

        # Try to find a proxy with matching scheme
        for _ in range(len(self.proxies)):
            proxy = self.proxies[self._proxy_index % len(self.proxies)]
            self._proxy_index += 1

            proxy_scheme = urlparse(proxy).scheme
            if proxy_scheme == url_scheme:
                return proxy

        # If no matching scheme found, return first proxy
        return self.proxies[0] if self.proxies else None

    def _select_proxy_deterministic(self, url: str) -> str:
        """Select proxy deterministically for testing."""
        parsed_url = urlparse(url)
        url_scheme = parsed_url.scheme

        # Find matching scheme proxy deterministically
        for proxy in self.proxies:
            proxy_scheme = urlparse(proxy).scheme
            if proxy_scheme == url_scheme:
                return proxy

        # Return first proxy if no scheme match
        return self.proxies[0]

    async def _get_domain_semaphore(self, domain: str) -> asyncio.Semaphore:
        """Get or create semaphore for domain."""
        try:
            current_loop = asyncio.get_running_loop()
            current_loop_id = id(current_loop)
        except RuntimeError:
            current_loop_id = None

        # Clear semaphores if we're in a different event loop
        if not hasattr(self, "_last_loop_id") or self._last_loop_id != current_loop_id:
            self._domain_semaphores.clear()
            self._last_loop_id = current_loop_id

        async with self._semaphore_lock:
            if domain not in self._domain_semaphores:
                self._domain_semaphores[domain] = asyncio.Semaphore(self.crawler_config.max_concurrency_per_domain)
            return self._domain_semaphores[domain]

    @property
    def robots_cache(self) -> Optional[RobotsTxtCache]:
        """Return robots.txt cache if initialized, else None."""
        if not self._is_initialized:
            return None

        # Cache the robots cache instance
        if not hasattr(self, "_robots_cache_instance"):
            assert self.session is not None
            self._robots_cache_instance = RobotsTxtCache(self.session)
        return self._robots_cache_instance

    async def initialize(self) -> None:
        """Initialize the HTTP client session."""
        if self.session is None:
            timeout = aiohttp.ClientTimeout(total=self.crawler_config.timeout)
            self.session = aiohttp.ClientSession(
                connector=self.connector, timeout=timeout, headers={"User-Agent": self.crawler_config.user_agent}
            )

            # Initialize robots cache if session is available
            if self.session is not None and not hasattr(self, "_robots_cache_instance"):
                self._robots_cache_instance = RobotsTxtCache(self.session)

            self._is_initialized = True

            # Always refresh the reference to `METRICS` – the autouse `metrics_reset`
            # fixture reloads the metrics module between tests, invalidating the objects
            # this module captured at import-time. Re-binding guarantees we observe on
            # the currently registered collectors.
            import importlib

            refreshed_metrics = importlib.import_module("quarrycore.observability.metrics").METRICS
            globals()["METRICS"] = refreshed_metrics

            logger.info("HTTP client session initialized")

    async def close(self) -> None:
        """Close the HTTP client and clean up resources."""
        if self.session:
            await self.session.close()
            self.session = None

        # Clear cached robots instance
        if hasattr(self, "_robots_cache_instance"):
            delattr(self, "_robots_cache_instance")

        # Clear in-flight per-domain semaphores
        self._domain_semaphores.clear()

        # Persist robots cache to disk if configured
        # TODO: Implement robots cache persistence when configuration is added

        self._is_initialized = False

        # Close connector if it's not a mock
        if hasattr(self.connector, "close") and not hasattr(self.connector, "_mock_name"):
            await self.connector.close()

        logger.info("HTTP client closed")

    async def __aenter__(self) -> "HttpClient":
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()

    async def _check_robots_txt(self, url: str) -> bool:
        """Check if URL is allowed by robots.txt."""
        try:
            if self.robots_cache is None:
                return True
            return await self.robots_cache.can_fetch(url, self.crawler_config.user_agent)
        except Exception as e:
            logger.warning("Failed to check robots.txt, allowing by default", url=url, error=str(e))
            return True

    async def _check_meta_robots(self, html_content: str) -> bool:
        """Check if content has meta robots noindex directive."""
        try:
            parser = HTMLParser(html_content)
            meta_robots = parser.css_first('meta[name="robots"]')
            if meta_robots:
                content = meta_robots.attributes.get("content")
                if content and "noindex" in content.lower():
                    return False
            return True
        except Exception as e:
            logger.debug("Error checking meta robots", error=str(e))
            return True

    async def _perform_request(self, url: str, timeout: float, proxy: Optional[str] = None) -> aiohttp.ClientResponse:
        """Perform the actual HTTP request."""
        if not self._is_initialized:
            raise RuntimeError("HTTP client not initialized")

        kwargs = {}
        if proxy:
            kwargs["proxy"] = proxy

        try:
            async with asyncio.timeout(timeout):
                assert self.session is not None
                response = await self.session.get(url, **kwargs)
                return response
        except asyncio.TimeoutError:
            # Re-raise as TimeoutError for consistent handling
            raise asyncio.TimeoutError(f"Request timed out after {timeout}s")

    async def _should_retry(self, response: aiohttp.ClientResponse, attempt: int, max_retries: int) -> bool:
        """Determine if request should be retried."""
        if attempt > max_retries:
            return False

        # Retry on server errors and rate limiting
        if response.status in {429, 503, 502, 504}:
            return True

        return False

    async def _calculate_backoff_delay(self, attempt: int) -> float:
        """Calculate exponential backoff delay with jitter."""
        base_delay = 2 ** (attempt - 1)  # 1s, 2s, 4s
        jitter = random.uniform(0.8, 1.2)  # ±20% jitter
        return base_delay * jitter

    async def fetch(self, url: str, *, timeout: float = 30.0, max_retries: Optional[int] = None) -> CrawlerResponse:
        """
        Fetch URL with robots.txt compliance, retries, and observability.

        Args:
            url: URL to fetch
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts (None = use config default)

        Returns:
            CrawlerResponse with status, headers, body, and timing info
        """
        if not self._is_initialized:
            raise RuntimeError("HTTP client not initialized. Call initialize() first.")

        start_time = time.time()

        # Parse URL safely, handle malformed URLs
        try:
            parsed_url = urlparse(url)
            domain = parsed_url.hostname or "unknown"
            # Check for completely malformed URLs
            if not parsed_url.scheme or not parsed_url.netloc:
                raise ValueError("Missing scheme or netloc")
        except (ValueError, AttributeError, TypeError) as e:
            logger.warning("Malformed URL", url=url, error=str(e))
            return CrawlerResponse(
                status=0,
                headers={},
                body=b"",
                start_ts=start_time,
                end_ts=time.time(),
                attempts=1,
                url=url,
                final_url=url,
            )

        if max_retries is None:
            max_retries = self.crawler_config.max_retries

        # Update metrics
        self._in_flight_requests += 1
        if "crawler_in_flight_requests" in METRICS:
            METRICS["crawler_in_flight_requests"].set(self._in_flight_requests)

        # Check domain backoff
        if await self.backoff_registry.is_in_cooldown(domain):
            logger.info("Domain in cooldown, skipping", domain=domain, url=url)
            end_time = time.time()

            # Update metrics
            self._in_flight_requests -= 1
            if "crawler_in_flight_requests" in METRICS:
                METRICS["crawler_in_flight_requests"].set(self._in_flight_requests)

            return CrawlerResponse(
                status=999,
                headers={},
                body=b"",
                start_ts=start_time,
                end_ts=end_time,
                attempts=0,
                url=url,
                final_url=url,
            )

        # Check robots.txt
        if self.crawler_config.respect_robots:
            if not await self._check_robots_txt(url):
                logger.info("Blocked by robots.txt", url=url)
                end_time = time.time()

                # Update metrics
                self._in_flight_requests -= 1
                if "crawler_in_flight_requests" in METRICS:
                    METRICS["crawler_in_flight_requests"].set(self._in_flight_requests)

                return CrawlerResponse(
                    status=999,
                    headers={},
                    body=b"",
                    start_ts=start_time,
                    end_ts=end_time,
                    attempts=0,
                    url=url,
                    final_url=url,
                )

        # Get domain semaphore for concurrency control
        semaphore = await self._get_domain_semaphore(domain)

        async with semaphore:
            # Track concurrent requests per domain
            self._domain_inflight[domain] = self._domain_inflight.get(domain, 0) + 1
            attempt = 0

            while attempt < max_retries + 1:
                attempt += 1

                try:
                    # Get proxy for this request
                    proxy = self._get_next_proxy(url)

                    # Perform request
                    response = await self._perform_request(url, timeout, proxy)

                    # Check if we should retry
                    if await self._should_retry(response, attempt, max_retries):
                        logger.info(
                            "Retrying request",
                            url=url,
                            status=response.status,
                            attempt=attempt,
                            max_retries=max_retries,
                        )

                        # Calculate backoff delay
                        delay = await self._calculate_backoff_delay(attempt)
                        await asyncio.sleep(delay)

                        await response.release()
                        continue

                    # Success or non-retryable error
                    content = await response.read()
                    end_time = time.time()

                    # Check meta robots if successful
                    if response.status == 200 and content:
                        try:
                            html_content = content.decode("utf-8", errors="ignore")
                            if not await self._check_meta_robots(html_content):
                                logger.info("Blocked by meta robots", url=url)
                                await response.release()

                                # Update metrics
                                self._in_flight_requests -= 1
                                if "crawler_in_flight_requests" in METRICS:
                                    METRICS["crawler_in_flight_requests"].set(self._in_flight_requests)

                                # Cleanup domain inflight tracking
                                self._domain_inflight[domain] = self._domain_inflight.get(domain, 0) - 1
                                if self._domain_inflight[domain] <= 0:
                                    del self._domain_inflight[domain]

                                return CrawlerResponse(
                                    status=999,
                                    headers={},
                                    body=b"",
                                    start_ts=start_time,
                                    end_ts=end_time,
                                    attempts=attempt,
                                    url=url,
                                    final_url=str(response.url),
                                )
                        except Exception as e:
                            logger.debug("Error checking meta robots", error=str(e))

                    # Record success/failure for backoff
                    if response.status < 500:
                        await self.backoff_registry.record_success(domain)
                    else:
                        await self.backoff_registry.record_failure(domain)

                    # Update metrics
                    status_class = f"{response.status // 100}xx"
                    if "crawler_responses_total" in METRICS:
                        METRICS["crawler_responses_total"].labels(status_class=status_class).inc()

                    if "crawler_fetch_latency_seconds" in METRICS:
                        METRICS["crawler_fetch_latency_seconds"].observe(end_time - start_time)

                    self._in_flight_requests -= 1
                    if "crawler_in_flight_requests" in METRICS:
                        METRICS["crawler_in_flight_requests"].set(self._in_flight_requests)

                    result = CrawlerResponse(
                        status=response.status,
                        headers=dict(response.headers),
                        body=content,
                        start_ts=start_time,
                        end_ts=end_time,
                        attempts=attempt,
                        url=url,
                        final_url=str(response.url),
                    )

                    # Cleanup domain inflight tracking
                    self._domain_inflight[domain] = self._domain_inflight.get(domain, 0) - 1
                    if self._domain_inflight[domain] <= 0:
                        del self._domain_inflight[domain]

                    await response.release()
                    return result

                except asyncio.TimeoutError as e:
                    logger.warning(
                        "Request timed out",
                        url=url,
                        attempt=attempt,
                        max_retries=max_retries,
                        timeout=timeout,
                        error=str(e),
                    )

                    if attempt < max_retries + 1:
                        delay = await self._calculate_backoff_delay(attempt)
                        await asyncio.sleep(delay)

                except Exception as e:
                    logger.warning("Request failed", url=url, attempt=attempt, max_retries=max_retries, error=str(e))

                    if attempt < max_retries + 1:
                        delay = await self._calculate_backoff_delay(attempt)
                        await asyncio.sleep(delay)

            # All retries exhausted
            await self.backoff_registry.record_failure(domain)

            # Update metrics
            if "crawler_domain_backoff_total" in METRICS:
                METRICS["crawler_domain_backoff_total"].inc()

            self._in_flight_requests -= 1
            if "crawler_in_flight_requests" in METRICS:
                METRICS["crawler_in_flight_requests"].set(self._in_flight_requests)

            end_time = time.time()

            # Cleanup domain inflight tracking
            self._domain_inflight[domain] = self._domain_inflight.get(domain, 0) - 1
            if self._domain_inflight[domain] <= 0:
                del self._domain_inflight[domain]

            # Return error response
            return CrawlerResponse(
                status=0,
                headers={},
                body=b"",
                start_ts=start_time,
                end_ts=end_time,
                attempts=attempt,
                url=url,
                final_url=url,
            )

    def get_stats(self) -> Dict[str, Any]:
        """Get current client statistics."""
        return {
            "in_flight_requests": self._in_flight_requests,
            "domain_semaphores": len(self._domain_semaphores),
            "proxy_count": len(self.proxies),
            "backoff_stats": self.backoff_registry.get_stats(),
            "domain_inflight": dict(self._domain_inflight),
        }
