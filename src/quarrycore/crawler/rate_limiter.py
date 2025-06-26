"""
Domain-Specific Rate Limiter with Adaptive Throttling

Implements intelligent rate limiting that adapts to server response patterns
and respects rate limit headers while optimizing for hardware capabilities.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Set
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


@dataclass
class DomainRateConfig:
    """Rate limiting configuration for a specific domain."""
    
    requests_per_second: float = 1.0
    burst_limit: int = 5
    last_request_time: float = 0.0
    current_burst: int = 0
    adaptive_delay: float = 1.0
    respect_retry_after: bool = True
    respect_rate_limit_headers: bool = True
    
    # Adaptive learning
    avg_response_time: float = 0.0
    error_rate: float = 0.0
    consecutive_errors: int = 0
    last_successful_request: float = 0.0


class DomainRateLimiter:
    """
    Intelligent rate limiter that adapts to server behavior.
    
    Features:
    - Per-domain rate limiting with burst capacity
    - Adaptive throttling based on response times and error rates
    - Respect for Retry-After and rate limit headers
    - Automatic adjustment for server performance
    - Hardware-aware optimization
    """
    
    def __init__(
        self,
        default_requests_per_second: float = 1.0,
        default_burst_limit: int = 5,
        max_requests_per_second: float = 10.0,
        min_requests_per_second: float = 0.1,
    ):
        self.default_rps = default_requests_per_second
        self.default_burst = default_burst_limit
        self.max_rps = max_requests_per_second
        self.min_rps = min_requests_per_second
        
        # Per-domain configurations
        self._domain_configs: Dict[str, DomainRateConfig] = {}
        
        # Global state
        self._forced_delays: Dict[str, float] = {}  # Domain -> end time of forced delay
        self._locks: Dict[str, asyncio.Lock] = {}
        
        logger.info(f"Rate limiter initialized with default {default_requests_per_second} RPS")
    
    def _get_domain_lock(self, domain: str) -> asyncio.Lock:
        """Get or create lock for domain."""
        if domain not in self._locks:
            self._locks[domain] = asyncio.Lock()
        return self._locks[domain]
    
    def _get_domain_config(self, domain: str) -> DomainRateConfig:
        """Get or create rate configuration for domain."""
        if domain not in self._domain_configs:
            self._domain_configs[domain] = DomainRateConfig(
                requests_per_second=self.default_rps,
                burst_limit=self.default_burst,
            )
            logger.debug(f"Created rate config for domain: {domain}")
        
        return self._domain_configs[domain]
    
    async def wait_for_domain(self, domain: str) -> float:
        """
        Wait for rate limit before making request to domain.
        
        Args:
            domain: Target domain
            
        Returns:
            Actual delay applied in seconds
        """
        domain_lock = self._get_domain_lock(domain)
        
        async with domain_lock:
            config = self._get_domain_config(domain)
            current_time = time.time()
            
            # Check for forced delays (from Retry-After headers)
            if domain in self._forced_delays:
                forced_end_time = self._forced_delays[domain]
                if current_time < forced_end_time:
                    delay = forced_end_time - current_time
                    logger.info(f"Respecting forced delay for {domain}: {delay:.2f}s")
                    await asyncio.sleep(delay)
                    return delay
                else:
                    # Forced delay expired
                    del self._forced_delays[domain]
            
            # Calculate delay based on rate limit
            time_since_last = current_time - config.last_request_time
            required_interval = 1.0 / config.requests_per_second
            
            # Check burst capacity
            if time_since_last >= required_interval:
                # Reset burst if enough time passed
                config.current_burst = 0
            
            if config.current_burst >= config.burst_limit:
                # Must wait for burst to reset
                delay = required_interval - time_since_last
                if delay > 0:
                    logger.debug(f"Burst limit reached for {domain}, waiting {delay:.2f}s")
                    await asyncio.sleep(delay)
                    config.current_burst = 0
                    delay_applied = delay
                else:
                    delay_applied = 0.0
            else:
                # Within burst limit
                delay_applied = 0.0
            
            # Apply adaptive delay if server is struggling
            if config.adaptive_delay > required_interval:
                additional_delay = config.adaptive_delay - required_interval
                if additional_delay > 0:
                    logger.debug(f"Applying adaptive delay for {domain}: {additional_delay:.2f}s")
                    await asyncio.sleep(additional_delay)
                    delay_applied += additional_delay
            
            # Update state
            config.last_request_time = time.time()
            config.current_burst += 1
            
            return delay_applied
    
    def update_from_response(self, domain: str, headers: Dict[str, str]) -> None:
        """
        Update rate limiting based on server response headers.
        
        Args:
            domain: Domain that was requested
            headers: Response headers from server
        """
        config = self._get_domain_config(domain)
        current_time = time.time()
        
        # Handle Retry-After header
        retry_after = headers.get("Retry-After") or headers.get("retry-after")
        if retry_after and config.respect_retry_after:
            try:
                delay_seconds = float(retry_after)
                self._forced_delays[domain] = current_time + delay_seconds
                logger.info(f"Server requested {delay_seconds}s delay for {domain}")
            except ValueError:
                # Retry-After might be a date, ignore for now
                pass
        
        # Handle rate limit headers
        if config.respect_rate_limit_headers:
            self._update_from_rate_limit_headers(domain, headers)
        
        # Update success metrics
        config.last_successful_request = current_time
        config.consecutive_errors = 0
        
        # Adaptive learning - increase rate on consistent success
        if config.error_rate < 0.1:  # Less than 10% error rate
            new_rps = min(config.requests_per_second * 1.1, self.max_rps)
            if new_rps != config.requests_per_second:
                logger.debug(f"Increasing rate for {domain}: {new_rps:.2f} RPS")
                config.requests_per_second = new_rps
                config.adaptive_delay = 1.0 / new_rps
    
    def _update_from_rate_limit_headers(self, domain: str, headers: Dict[str, str]) -> None:
        """Update rate limits based on standard rate limit headers."""
        config = self._get_domain_config(domain)
        
        # Common rate limit header patterns
        rate_limit_headers = [
            ("X-RateLimit-Limit", "X-RateLimit-Remaining", "X-RateLimit-Reset"),
            ("X-Rate-Limit-Limit", "X-Rate-Limit-Remaining", "X-Rate-Limit-Reset"),
            ("RateLimit-Limit", "RateLimit-Remaining", "RateLimit-Reset"),
        ]
        
        for limit_header, remaining_header, reset_header in rate_limit_headers:
            limit = headers.get(limit_header)
            remaining = headers.get(remaining_header)
            reset_time = headers.get(reset_header)
            
            if limit and remaining:
                try:
                    limit_value = int(limit)
                    remaining_value = int(remaining)
                    
                    # Calculate current utilization
                    utilization = (limit_value - remaining_value) / limit_value
                    
                    # Adjust rate based on remaining quota
                    if remaining_value < limit_value * 0.2:  # Less than 20% remaining
                        # Slow down significantly
                        new_rps = max(config.requests_per_second * 0.5, self.min_rps)
                        logger.info(f"Rate limit warning for {domain}, reducing to {new_rps:.2f} RPS")
                        config.requests_per_second = new_rps
                        config.adaptive_delay = 1.0 / new_rps
                    
                    # If reset time is provided, calculate optimal rate
                    if reset_time:
                        try:
                            reset_timestamp = float(reset_time)
                            current_time = time.time()
                            time_until_reset = reset_timestamp - current_time
                            
                            if time_until_reset > 0 and remaining_value > 0:
                                # Calculate optimal rate to use remaining quota
                                optimal_rps = remaining_value / time_until_reset
                                config.requests_per_second = min(optimal_rps, self.max_rps)
                                logger.debug(f"Calculated optimal rate for {domain}: {optimal_rps:.2f} RPS")
                        
                        except ValueError:
                            pass
                    
                    break  # Found valid headers
                
                except ValueError:
                    continue
    
    def record_error(self, domain: str, error_type: str = "generic") -> None:
        """
        Record an error for adaptive rate limiting.
        
        Args:
            domain: Domain where error occurred
            error_type: Type of error (timeout, 5xx, etc.)
        """
        config = self._get_domain_config(domain)
        config.consecutive_errors += 1
        
        # Update error rate (simple exponential moving average)
        config.error_rate = config.error_rate * 0.9 + 0.1
        
        # Adaptive backoff based on error type and frequency
        if error_type in ("timeout", "connection_error"):
            # Aggressive backoff for connection issues
            new_rps = max(config.requests_per_second * 0.3, self.min_rps)
        elif error_type.startswith("5"):  # 5xx server errors
            # Moderate backoff for server errors
            new_rps = max(config.requests_per_second * 0.6, self.min_rps)
        elif config.consecutive_errors > 5:
            # Gradual backoff for repeated errors
            new_rps = max(config.requests_per_second * 0.8, self.min_rps)
        else:
            return  # No adjustment needed
        
        if new_rps != config.requests_per_second:
            logger.warning(
                f"Reducing rate for {domain} due to {error_type}: {new_rps:.2f} RPS "
                f"(consecutive errors: {config.consecutive_errors})"
            )
            config.requests_per_second = new_rps
            config.adaptive_delay = 1.0 / new_rps
    
    def record_response_time(self, domain: str, response_time_ms: float) -> None:
        """
        Record response time for adaptive optimization.
        
        Args:
            domain: Domain that was requested
            response_time_ms: Response time in milliseconds
        """
        config = self._get_domain_config(domain)
        
        # Update average response time (exponential moving average)
        if config.avg_response_time == 0:
            config.avg_response_time = response_time_ms
        else:
            config.avg_response_time = config.avg_response_time * 0.9 + response_time_ms * 0.1
        
        # Adjust rate based on response time
        if response_time_ms > 5000:  # Very slow response (>5s)
            new_rps = max(config.requests_per_second * 0.7, self.min_rps)
            logger.debug(f"Slow response from {domain} ({response_time_ms:.0f}ms), reducing rate")
            config.requests_per_second = new_rps
            config.adaptive_delay = 1.0 / new_rps
        elif response_time_ms < 500 and config.error_rate < 0.05:  # Fast and reliable
            new_rps = min(config.requests_per_second * 1.05, self.max_rps)
            if new_rps != config.requests_per_second:
                logger.debug(f"Fast response from {domain} ({response_time_ms:.0f}ms), increasing rate")
                config.requests_per_second = new_rps
                config.adaptive_delay = 1.0 / new_rps
    
    def get_domain_stats(self, domain: str) -> Dict[str, Any]:
        """Get statistics for a specific domain."""
        if domain not in self._domain_configs:
            return {"exists": False}
        
        config = self._domain_configs[domain]
        current_time = time.time()
        
        return {
            "exists": True,
            "requests_per_second": config.requests_per_second,
            "burst_limit": config.burst_limit,
            "current_burst": config.current_burst,
            "adaptive_delay": config.adaptive_delay,
            "avg_response_time": config.avg_response_time,
            "error_rate": config.error_rate,
            "consecutive_errors": config.consecutive_errors,
            "time_since_last_request": current_time - config.last_request_time,
            "forced_delay_remaining": max(0, self._forced_delays.get(domain, 0) - current_time),
        }
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all domains."""
        return {
            domain: self.get_domain_stats(domain)
            for domain in self._domain_configs.keys()
        }
    
    def reset_domain(self, domain: str) -> None:
        """Reset rate limiting state for a domain."""
        if domain in self._domain_configs:
            del self._domain_configs[domain]
        if domain in self._forced_delays:
            del self._forced_delays[domain]
        if domain in self._locks:
            del self._locks[domain]
        logger.info(f"Reset rate limiting for domain: {domain}")
    
    def adapt_to_hardware(self, cpu_cores: int, available_memory_gb: float) -> None:
        """Adapt rate limiting to hardware capabilities."""
        # More powerful hardware can handle higher rates
        memory_factor = min(2.0, available_memory_gb / 8.0)
        cpu_factor = min(2.0, cpu_cores / 4.0)
        
        hardware_multiplier = (memory_factor + cpu_factor) / 2.0
        
        # Adjust default rates
        new_default_rps = self.default_rps * hardware_multiplier
        new_max_rps = self.max_rps * hardware_multiplier
        
        self.default_rps = min(new_default_rps, 20.0)  # Cap at 20 RPS
        self.max_rps = min(new_max_rps, 50.0)  # Cap at 50 RPS
        
        logger.info(
            f"Adapted rate limiter to hardware: "
            f"default={self.default_rps:.2f} RPS, max={self.max_rps:.2f} RPS"
        ) 