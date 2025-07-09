"""
Adaptive Web Crawler with HTTP/2 and Hardware-Aware Optimization

This module implements the core crawler that adapts to hardware capabilities,
from Raspberry Pi constraints to high-end workstation performance, while
maintaining ethical scraping practices and robust error handling.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple
from urllib.parse import urlparse
from uuid import UUID, uuid4

import httpx
from httpx import AsyncClient

from ..protocols import (
    CrawlResult,
    ErrorInfo,
    ErrorSeverity,
    HardwareCapabilities,
    HardwareType,
    PerformanceMetrics,
    ProcessingStatus,
    create_correlation_id,
)
from .circuit_breaker import CircuitBreaker
from .http_client import HttpClient
from .rate_limiter import DomainRateLimiter
from .robots_parser import RobotsCache
from .user_agents import UserAgentRotator

logger = logging.getLogger(__name__)


@dataclass
class ConnectionPoolConfig:
    """Configuration for HTTP connection pooling."""

    max_connections_per_host: int = 20
    max_keepalive_connections: int = 100
    keepalive_expiry: float = 30.0
    http2: bool = True
    verify_ssl: bool = True
    timeout: float = 30.0
    max_redirects: int = 5


@dataclass
class AdaptiveConfig:
    """Adaptive configuration based on hardware capabilities."""

    max_concurrent_requests: int = 50
    max_concurrent_per_domain: int = 5
    request_delay_seconds: float = 1.0
    bandwidth_limit_mbps: Optional[float] = None
    use_playwright_fallback: bool = True
    stream_large_responses: bool = True
    stream_threshold_mb: float = 10.0

    # Retry configuration
    max_retries: int = 3
    retry_base_delay: float = 1.0
    retry_max_delay: float = 60.0
    retry_jitter: bool = True

    # Circuit breaker configuration
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: float = 60.0
    circuit_breaker_reset_timeout: float = 300.0


class AdaptiveCrawler:
    """
    Production-grade adaptive web crawler with HTTP/2 support.

    Features:
    - Adaptive concurrency based on hardware capabilities
    - HTTP/2 persistent connections with smart pooling
    - Circuit breaker pattern for failing domains
    - Exponential backoff with jitter
    - Robots.txt compliance with caching
    - User-agent rotation and bandwidth throttling
    - ETag/Last-Modified support for efficient caching
    - Playwright fallback for JavaScript-heavy sites
    """

    _last_loop_id: Optional[int] = None

    def __init__(
        self,
        hardware_caps: Optional[HardwareCapabilities] = None,
        connection_config: Optional[ConnectionPoolConfig] = None,
        adaptive_config: Optional[AdaptiveConfig] = None,
    ):
        self.hardware_caps = hardware_caps
        self.connection_config = connection_config or ConnectionPoolConfig()
        self.adaptive_config = adaptive_config or AdaptiveConfig()

        # Core components
        self._client: Optional[AsyncClient] = None
        self._playwright_client: Optional[Any] = None  # Lazy loaded

        # Adaptive components
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._domain_semaphores: Dict[str, asyncio.Semaphore] = {}
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._rate_limiter = DomainRateLimiter()
        self._robots_cache = RobotsCache()
        self._user_agent_rotator = UserAgentRotator()

        # Performance tracking
        self._start_time = time.time()
        self._total_requests = 0
        self._successful_requests = 0
        self._failed_requests = 0
        self._total_bytes_downloaded = 0
        self._response_cache: Dict[str, Tuple[str, datetime, bytes]] = {}

        # Adaptive configuration
        if hardware_caps:
            self._adapt_to_hardware(hardware_caps)

        logger.info("AdaptiveCrawler initialized with adaptive configuration")

    def _adapt_to_hardware(self, capabilities: HardwareCapabilities) -> None:
        """Adapt crawler configuration to hardware capabilities."""
        # Calculate adaptive concurrency: CPU cores × 5, with reasonable bounds
        base_concurrency = max(4, capabilities.cpu_cores * 5)

        # Scale based on available memory (more memory = more concurrent connections)
        # Safety guard: if available_memory_gb is None/0, use total_memory_gb * 0.75 as fallback
        available_memory = capabilities.available_memory_gb
        if available_memory is None or available_memory <= 0:
            available_memory = capabilities.total_memory_gb * 0.75
            logger.warning(
                f"available_memory_gb not set or invalid ({capabilities.available_memory_gb}), using fallback: {available_memory}GB"
            )

        memory_factor = min(2.0, available_memory / 8.0)

        # Scale based on hardware type
        hardware_multipliers = {
            HardwareType.RASPBERRY_PI: 0.3,
            HardwareType.LAPTOP: 0.7,
            HardwareType.WORKSTATION: 1.0,
            HardwareType.SERVER: 1.5,
            HardwareType.CLOUD: 2.0,
        }

        hardware_factor = hardware_multipliers.get(capabilities.hardware_type, 1.0)

        # Calculate final concurrency
        adaptive_concurrency = int(base_concurrency * memory_factor * hardware_factor)
        self.adaptive_config.max_concurrent_requests = max(1, min(adaptive_concurrency, 200))  # Ensure at least 1

        # Adjust per-domain limits
        self.adaptive_config.max_concurrent_per_domain = max(
            1, self.adaptive_config.max_concurrent_requests // 10
        )  # Ensure at least 1

        # Adjust request delays based on hardware
        if capabilities.hardware_type == HardwareType.RASPBERRY_PI:
            self.adaptive_config.request_delay_seconds = 2.0
            self.adaptive_config.stream_threshold_mb = 5.0
        elif capabilities.hardware_type == HardwareType.WORKSTATION:
            self.adaptive_config.request_delay_seconds = 0.5
            self.adaptive_config.stream_threshold_mb = 20.0

        # Bandwidth limiting for Pi
        if capabilities.hardware_type == HardwareType.RASPBERRY_PI:
            self.adaptive_config.bandwidth_limit_mbps = 10.0

        logger.info(
            f"Adapted crawler for {capabilities.hardware_type.value}: "
            f"concurrency={self.adaptive_config.max_concurrent_requests}, "
            f"delay={self.adaptive_config.request_delay_seconds}s"
        )

    async def __aenter__(self) -> "AdaptiveCrawler":
        """Async context manager entry."""
        await self._initialize_client()
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[Exception],
        exc_tb: Optional[object],
    ) -> None:
        """Async context manager exit with cleanup."""
        await self._cleanup()

    async def _initialize_client(self) -> None:
        """Initialize HTTP client with optimized configuration."""
        # Create semaphore for global concurrency control
        self._semaphore = asyncio.Semaphore(self.adaptive_config.max_concurrent_requests)

        # Configure HTTP client with connection pooling
        limits = httpx.Limits(
            max_connections=self.connection_config.max_connections_per_host,
            max_keepalive_connections=self.connection_config.max_keepalive_connections,
            keepalive_expiry=self.connection_config.keepalive_expiry,
        )

        timeout = httpx.Timeout(
            connect=10.0,
            read=self.connection_config.timeout,
            write=10.0,
            pool=2.0,
        )

        # Create client with HTTP/2 support
        self._client = AsyncClient(
            limits=limits,
            timeout=timeout,
            http2=self.connection_config.http2,
            verify=self.connection_config.verify_ssl,
            follow_redirects=True,
            max_redirects=self.connection_config.max_redirects,
        )

        logger.info("HTTP client initialized with HTTP/2 and connection pooling")

    async def _cleanup(self) -> None:
        """Clean up resources."""
        if self._client:
            await self._client.aclose()

        if self._playwright_client:
            await self._playwright_client.close()

        # Clear semaphores
        self._domain_semaphores.clear()

        logger.info("AdaptiveCrawler cleanup completed")

    def _get_domain_semaphore(self, domain: str) -> asyncio.Semaphore:
        """Get or create domain-specific semaphore for rate limiting."""
        try:
            current_loop = asyncio.get_running_loop()
            current_loop_id = id(current_loop)
        except RuntimeError:
            current_loop_id = None

        # Clear semaphores if we're in a different event loop
        if not hasattr(self, "_last_loop_id") or self._last_loop_id != current_loop_id:
            self._domain_semaphores.clear()
            self._last_loop_id = current_loop_id

        # Create semaphore if needed
        if domain not in self._domain_semaphores:
            self._domain_semaphores[domain] = asyncio.Semaphore(self.adaptive_config.max_concurrent_per_domain)
        else:
            pass

        semaphore = self._domain_semaphores[domain]
        return semaphore

    def _get_circuit_breaker(self, domain: str) -> CircuitBreaker:
        """Get or create circuit breaker for domain."""
        if domain not in self._circuit_breakers:
            self._circuit_breakers[domain] = CircuitBreaker(
                failure_threshold=self.adaptive_config.circuit_breaker_threshold,
                timeout_duration=self.adaptive_config.circuit_breaker_timeout,
                reset_timeout=self.adaptive_config.circuit_breaker_reset_timeout,
            )
        return self._circuit_breakers[domain]

    async def _calculate_delay_with_jitter(self, attempt: int) -> float:
        """Calculate exponential backoff delay with jitter."""
        base_delay = self.adaptive_config.retry_base_delay
        max_delay = self.adaptive_config.retry_max_delay

        # Exponential backoff: base * (2 ^ attempt)
        delay = min(base_delay * (2**attempt), max_delay)

        # Add jitter to prevent thundering herd
        if self.adaptive_config.retry_jitter:
            jitter = random.uniform(0.1, 0.3) * delay
            delay += jitter

        return delay

    async def _check_response_cache(self, url: str, headers: Dict[str, str]) -> Optional[CrawlResult]:
        """Check response cache for ETag/Last-Modified optimization."""
        if url not in self._response_cache:
            return None

        etag, last_modified_time, cached_content = self._response_cache[url]

        # Check if cache is still valid (within 1 hour)
        if datetime.utcnow() - last_modified_time > timedelta(hours=1):
            del self._response_cache[url]
            return None

        # Add conditional headers
        if etag:
            headers["If-None-Match"] = etag
        headers["If-Modified-Since"] = last_modified_time.strftime("%a, %d %b %Y %H:%M:%S GMT")

        return None  # Continue with conditional request

    async def _update_response_cache(self, url: str, response: httpx.Response, content: bytes) -> None:
        """Update response cache with ETag and Last-Modified data."""
        etag = response.headers.get("ETag")
        last_modified = response.headers.get("Last-Modified")

        if etag or last_modified:
            self._response_cache[url] = (etag or "", datetime.utcnow(), content)

    async def _stream_large_response(self, response: httpx.Response) -> bytes:
        """Stream large responses to manage memory efficiently."""
        content_length = response.headers.get("Content-Length")

        # Check if response should be streamed
        should_stream = (
            self.adaptive_config.stream_large_responses
            and content_length
            and int(content_length) > self.adaptive_config.stream_threshold_mb * 1024 * 1024
        )

        if should_stream:
            # Stream response in chunks
            content_chunks = []
            async for chunk in response.aiter_bytes(chunk_size=1024 * 1024):  # 1MB chunks
                content_chunks.append(chunk)

                # Bandwidth throttling for Raspberry Pi
                if self.adaptive_config.bandwidth_limit_mbps:
                    await asyncio.sleep(0.01)  # Small delay for bandwidth control

            return b"".join(content_chunks)
        else:
            return await response.aread()

    async def crawl_url(
        self,
        url: str,
        *,
        headers: Optional[Dict[str, str]] = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        respect_robots: bool = True,
    ) -> CrawlResult:
        """
        Crawl a single URL with comprehensive error handling and optimization.

        Args:
            url: Target URL to crawl
            headers: Optional custom headers
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            respect_robots: Whether to check robots.txt

        Returns:
            CrawlResult with performance metrics and error tracking
        """
        request_id = uuid4()
        correlation_id = create_correlation_id()
        start_time = time.time()
        parsed_url = urlparse(url)
        domain = parsed_url.netloc

        # Initialize result object
        result = CrawlResult(
            request_id=request_id,
            url=url,
            final_url=url,
            timestamp=datetime.utcnow(),
        )

        try:
            # Check robots.txt compliance
            if respect_robots:
                robots_allowed = await self._robots_cache.is_allowed(url, "*")
                result.robots_allowed = robots_allowed
                if not robots_allowed:
                    result.status = ProcessingStatus.SKIPPED
                    result.warnings.append("Blocked by robots.txt")
                    return result

            # Get circuit breaker for domain
            circuit_breaker = self._get_circuit_breaker(domain)

            # Check circuit breaker state
            if not await circuit_breaker.can_execute():
                result.status = ProcessingStatus.FAILED
                result.errors.append(
                    ErrorInfo(
                        error_type="CircuitBreakerOpen",
                        error_message=f"Circuit breaker open for domain {domain}",
                        severity=ErrorSeverity.HIGH,
                        correlation_id=correlation_id,
                    )
                )
                return result

            # Perform crawl with retries
            for attempt in range(max_retries + 1):
                try:
                    crawl_result = await self._perform_single_crawl(
                        url=url,
                        headers=headers,
                        timeout=timeout,
                        domain=domain,
                        correlation_id=correlation_id,
                    )

                    # Success - mark circuit breaker and return
                    await circuit_breaker.record_success()

                    # Calculate performance metrics
                    total_time = (time.time() - start_time) * 1000
                    crawl_result.performance = PerformanceMetrics(
                        total_duration_ms=total_time,
                        download_duration_ms=crawl_result.performance.download_duration_ms,
                        parsing_duration_ms=0.0,
                        extraction_duration_ms=0.0,
                        processing_duration_ms=total_time,
                        bytes_downloaded=len(crawl_result.content),
                        bytes_processed=len(crawl_result.content),
                        documents_per_second=(1.0 / (total_time / 1000) if total_time > 0 else 0.0),
                        peak_memory_mb=0.0,  # Would need memory profiling
                        avg_cpu_percent=0.0,  # Would need CPU monitoring
                        success_rate=1.0,
                        retry_count=attempt,
                    )

                    self._successful_requests += 1
                    self._total_bytes_downloaded += len(crawl_result.content)

                    return crawl_result

                except Exception as e:
                    error_info = ErrorInfo(
                        error_type=type(e).__name__,
                        error_message=str(e),
                        severity=(ErrorSeverity.MEDIUM if attempt < max_retries else ErrorSeverity.HIGH),
                        correlation_id=correlation_id,
                        retry_count=attempt,
                        max_retries=max_retries,
                        is_retryable=attempt < max_retries,
                    )

                    result.errors.append(error_info)

                    # Record failure in circuit breaker
                    await circuit_breaker.record_failure()

                    if attempt < max_retries:
                        # Wait before retry with exponential backoff
                        delay = await self._calculate_delay_with_jitter(attempt)
                        await asyncio.sleep(delay)
                        logger.warning(f"Retry {attempt + 1}/{max_retries} for {url} after {delay:.2f}s delay")
                    else:
                        # Max retries exceeded
                        result.status = ProcessingStatus.FAILED
                        self._failed_requests += 1
                        logger.error(f"Failed to crawl {url} after {max_retries} retries: {e}")

        except Exception as e:
            # Unexpected error
            result.status = ProcessingStatus.FAILED
            result.errors.append(
                ErrorInfo(
                    error_type=type(e).__name__,
                    error_message=str(e),
                    severity=ErrorSeverity.CRITICAL,
                    correlation_id=correlation_id,
                )
            )
            self._failed_requests += 1
            logger.error(f"Unexpected error crawling {url}: {e}")

        finally:
            self._total_requests += 1

        return result

    async def _perform_single_crawl(
        self,
        url: str,
        headers: Optional[Dict[str, str]],
        timeout: float,
        domain: str,
        correlation_id: UUID,
    ) -> CrawlResult:
        """Perform a single crawl attempt with all optimizations."""
        request_headers = headers.copy() if headers else {}

        # Add user agent rotation
        user_agent = self._user_agent_rotator.get_random_user_agent()
        request_headers["User-Agent"] = user_agent

        # Check response cache for conditional requests
        await self._check_response_cache(url, request_headers)

        # Use semaphores for concurrency control
        if self._semaphore is None:
            raise RuntimeError("Crawler not initialized. Use async context manager.")

        async with self._semaphore:  # Global concurrency limit
            async with self._get_domain_semaphore(domain):  # Per-domain concurrency limit
                # Apply rate limiting
                await self._rate_limiter.wait_for_domain(domain)

                # Make HTTP request
                download_start = time.time()

                if self._client is None:
                    raise RuntimeError("HTTP client not initialized.")
                response = await self._client.get(
                    url,
                    headers=request_headers,
                    timeout=timeout,
                )

                download_time = (time.time() - download_start) * 1000

                # Handle 304 Not Modified
                if response.status_code == 304:
                    # Return cached content
                    if url in self._response_cache:
                        _, _, cached_content = self._response_cache[url]
                        return CrawlResult(
                            url=url,
                            final_url=str(response.url),
                            status_code=304,
                            content=cached_content,
                            headers=dict(response.headers),
                            status=ProcessingStatus.COMPLETED,
                            user_agent=user_agent,
                            content_type=response.headers.get("Content-Type", ""),
                            content_length=len(cached_content),
                            is_valid=True,
                            performance=PerformanceMetrics(
                                total_duration_ms=download_time,
                                download_duration_ms=download_time,
                                parsing_duration_ms=0.0,
                                extraction_duration_ms=0.0,
                                processing_duration_ms=download_time,
                                bytes_downloaded=0,  # From cache
                                bytes_processed=len(cached_content),
                                documents_per_second=0.0,
                                peak_memory_mb=0.0,
                                avg_cpu_percent=0.0,
                                cache_hit_ratio=1.0,
                            ),
                        )

                # Stream large responses
                content = await self._stream_large_response(response)

                # Update response cache
                await self._update_response_cache(url, response, content)

                # Update rate limiter based on response
                self._rate_limiter.update_from_response(domain, dict(response.headers))

                # Create successful result
                result = CrawlResult(
                    url=url,
                    final_url=str(response.url),
                    status_code=response.status_code,
                    content=content,
                    headers=dict(response.headers),
                    status=ProcessingStatus.COMPLETED,
                    user_agent=user_agent,
                    content_type=response.headers.get("Content-Type", ""),
                    content_encoding=response.headers.get("Content-Encoding", ""),
                    content_length=len(content),
                    is_valid=200 <= response.status_code < 300,
                    performance=PerformanceMetrics(
                        total_duration_ms=download_time,
                        download_duration_ms=download_time,
                        parsing_duration_ms=0.0,
                        extraction_duration_ms=0.0,
                        processing_duration_ms=download_time,
                        bytes_downloaded=len(content),
                        bytes_processed=len(content),
                        documents_per_second=0.0,
                        peak_memory_mb=0.0,
                        avg_cpu_percent=0.0,
                    ),
                )

                return result

    async def crawl_batch(
        self,
        urls: List[str],
        *,
        concurrency: Optional[int] = None,
        rate_limit: Optional[float] = None,
        hardware_caps: Optional[HardwareCapabilities] = None,
    ) -> AsyncIterator[CrawlResult]:
        """
        Crawl multiple URLs with adaptive concurrency and rate limiting.

        Args:
            urls: List of URLs to crawl
            concurrency: Override default concurrency
            rate_limit: Override default rate limit
            hardware_caps: Hardware capabilities for adaptation

        Yields:
            CrawlResult objects as they complete
        """
        if hardware_caps and hardware_caps != self.hardware_caps:
            await self.adapt_to_hardware(hardware_caps)

        # Use custom concurrency if provided
        effective_concurrency = concurrency or self.adaptive_config.max_concurrent_requests
        semaphore = asyncio.Semaphore(effective_concurrency)

        async def crawl_with_semaphore(url: str) -> CrawlResult:
            async with semaphore:
                try:
                    result = await self.crawl_url(url)
                    return result
                except Exception:
                    raise

        # Create tasks for all URLs
        tasks = [crawl_with_semaphore(url) for url in urls]

        # Yield results as they complete
        for coro in asyncio.as_completed(tasks):
            try:
                result = await coro
                yield result
            except Exception as e:
                # Yield error result
                error_result = CrawlResult(
                    url="unknown",
                    status=ProcessingStatus.FAILED,
                    errors=[
                        ErrorInfo(
                            error_type=type(e).__name__,
                            error_message=str(e),
                            severity=ErrorSeverity.HIGH,
                        )
                    ],
                )
                yield error_result

    async def get_robots_txt(self, domain: str) -> Dict[str, Any]:
        """Fetch and parse robots.txt for domain."""
        # Since RobotsCache doesn't have get_robots_info, create a simple implementation
        try:
            robots_content = await self._robots_cache._fetch_robots_txt(domain)
            return {
                "domain": domain,
                "has_robots_txt": robots_content is not None,
                "content": robots_content or "",
                "size": len(robots_content) if robots_content else 0,
            }
        except Exception as e:
            return {
                "domain": domain,
                "has_robots_txt": False,
                "error": str(e),
                "content": "",
                "size": 0,
            }

    async def adapt_to_hardware(self, capabilities: HardwareCapabilities) -> None:
        """Adapt crawler settings based on hardware capabilities."""
        self.hardware_caps = capabilities
        self._adapt_to_hardware(capabilities)

        # Recreate semaphore with new limits
        if self._semaphore:
            self._semaphore = asyncio.Semaphore(self.adaptive_config.max_concurrent_requests)

        logger.info(f"Adapted crawler to hardware: {capabilities.hardware_type.value}")

    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics and statistics."""
        runtime = time.time() - self._start_time

        return {
            "runtime_seconds": runtime,
            "total_requests": self._total_requests,
            "successful_requests": self._successful_requests,
            "failed_requests": self._failed_requests,
            "success_rate": (self._successful_requests / self._total_requests if self._total_requests > 0 else 0.0),
            "requests_per_second": (self._total_requests / runtime if runtime > 0 else 0.0),
            "total_bytes_downloaded": self._total_bytes_downloaded,
            "avg_bytes_per_request": (
                self._total_bytes_downloaded / self._successful_requests if self._successful_requests > 0 else 0.0
            ),
            "active_domains": len(self._domain_semaphores),
            "circuit_breakers": {domain: cb.get_state() for domain, cb in self._circuit_breakers.items()},
            "cache_size": len(self._response_cache),
            "configuration": {
                "max_concurrent": self.adaptive_config.max_concurrent_requests,
                "max_per_domain": self.adaptive_config.max_concurrent_per_domain,
                "request_delay": self.adaptive_config.request_delay_seconds,
            },
        }

    async def _calculate_complexity_score(self, response_data: Dict[str, Any]) -> float:
        """Calculate content complexity score for adaptive crawling."""
        try:
            content_size = response_data.get("content_length", 0)
            js_count = response_data.get("script_tags", 0)
            form_count = response_data.get("forms", 0)

            # Base complexity from content size
            size_factor = min(1.0, content_size / 1_000_000)  # Normalize to 1MB

            # JavaScript complexity
            js_factor = min(1.0, js_count / 10)

            # Form complexity
            form_factor = min(1.0, form_count / 5)

            # Combine factors
            complexity = (size_factor * 0.4) + (js_factor * 0.4) + (form_factor * 0.2)

            return float(complexity)

        except Exception as e:
            logger.warning(f"Complexity calculation error: {e}")
            return 0.5  # Default complexity
