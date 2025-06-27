"""
Circuit Breaker Pattern Implementation for Domain Failure Management

Prevents cascading failures by temporarily stopping requests to failing domains
and automatically attempting recovery after a cooldown period.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, blocking requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""

    failure_threshold: int = 5  # Failures before opening
    timeout_duration: float = 60.0  # How long to stay open (seconds)
    reset_timeout: float = 300.0  # Full reset after this time
    success_threshold: int = 3  # Successes needed in half-open to close


class CircuitBreaker:
    """
    Circuit breaker for managing failing domains.

    Implements the circuit breaker pattern to prevent overwhelming
    failing services and provide automatic recovery detection.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        timeout_duration: float = 60.0,
        reset_timeout: float = 300.0,
        success_threshold: int = 3,
    ):
        self.config = CircuitBreakerConfig(
            failure_threshold=failure_threshold,
            timeout_duration=timeout_duration,
            reset_timeout=reset_timeout,
            success_threshold=success_threshold,
        )

        # State tracking
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._last_success_time: Optional[float] = None

        # Lock for thread safety
        self._lock = asyncio.Lock()

        logger.debug(f"Circuit breaker initialized with threshold {failure_threshold}")

    async def can_execute(self) -> bool:
        """Check if requests can be executed through this circuit breaker."""
        async with self._lock:
            current_time = time.time()

            if self._state == CircuitState.CLOSED:
                return True

            elif self._state == CircuitState.OPEN:
                # Check if enough time has passed to attempt recovery
                if self._last_failure_time and current_time - self._last_failure_time >= self.config.timeout_duration:
                    logger.info("Circuit breaker transitioning to HALF_OPEN for recovery test")
                    self._state = CircuitState.HALF_OPEN
                    self._success_count = 0
                    return True

                return False

            else:  # self._state == CircuitState.HALF_OPEN
                # Allow limited requests to test recovery
                return True

    async def record_success(self) -> None:
        """Record a successful operation."""
        async with self._lock:
            current_time = time.time()
            self._last_success_time = current_time

            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                logger.debug(f"Circuit breaker success count: {self._success_count}")

                # Check if we have enough successes to close the circuit
                if self._success_count >= self.config.success_threshold:
                    logger.info("Circuit breaker closing - service recovered")
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    self._success_count = 0

            elif self._state == CircuitState.CLOSED:
                # Reset failure count on success
                if self._failure_count > 0:
                    self._failure_count = max(0, self._failure_count - 1)

    async def record_failure(self) -> None:
        """Record a failed operation."""
        async with self._lock:
            current_time = time.time()
            self._last_failure_time = current_time

            if self._state in (CircuitState.CLOSED, CircuitState.HALF_OPEN):
                self._failure_count += 1
                logger.debug(f"Circuit breaker failure count: {self._failure_count}")

                # Check if we need to open the circuit
                if self._failure_count >= self.config.failure_threshold:
                    logger.warning(f"Circuit breaker opening due to {self._failure_count} failures")
                    self._state = CircuitState.OPEN
                    self._success_count = 0

    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state for monitoring."""
        return {
            "state": self._state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "last_failure_time": self._last_failure_time,
            "last_success_time": self._last_success_time,
            "can_execute": asyncio.create_task(self.can_execute()),
        }

    async def reset(self) -> None:
        """Manually reset the circuit breaker to closed state."""
        async with self._lock:
            logger.info("Circuit breaker manually reset")
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._last_failure_time = None
            self._last_success_time = None

    async def force_open(self) -> None:
        """Manually force the circuit breaker to open state."""
        async with self._lock:
            logger.warning("Circuit breaker manually forced open")
            self._state = CircuitState.OPEN
            self._last_failure_time = time.time()


class CircuitBreakerManager:
    """
    Manages multiple circuit breakers for different domains/services.
    """

    def __init__(self, default_config: Optional[CircuitBreakerConfig] = None):
        self.default_config = default_config or CircuitBreakerConfig()
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._lock = asyncio.Lock()

    async def get_circuit_breaker(self, domain: str) -> CircuitBreaker:
        """Get or create circuit breaker for a domain."""
        if domain not in self._circuit_breakers:
            async with self._lock:
                if domain not in self._circuit_breakers:
                    self._circuit_breakers[domain] = CircuitBreaker(
                        failure_threshold=self.default_config.failure_threshold,
                        timeout_duration=self.default_config.timeout_duration,
                        reset_timeout=self.default_config.reset_timeout,
                        success_threshold=self.default_config.success_threshold,
                    )
                    logger.debug(f"Created circuit breaker for domain: {domain}")

        return self._circuit_breakers[domain]

    async def get_all_states(self) -> Dict[str, Dict[str, Any]]:
        """Get states of all circuit breakers for monitoring."""
        states = {}
        for domain, cb in self._circuit_breakers.items():
            states[domain] = cb.get_state()
        return states

    async def reset_all(self) -> None:
        """Reset all circuit breakers."""
        for cb in self._circuit_breakers.values():
            await cb.reset()
        logger.info("All circuit breakers reset")

    async def cleanup_idle(self, max_idle_time: float = 3600.0) -> int:
        """Clean up circuit breakers that haven't been used recently."""
        current_time = time.time()
        removed_count = 0

        domains_to_remove = []
        for domain, cb in self._circuit_breakers.items():
            # Remove if no recent activity
            if (cb._last_failure_time is None and cb._last_success_time is None) or (
                cb._last_success_time and current_time - cb._last_success_time > max_idle_time
            ):
                domains_to_remove.append(domain)

        for domain in domains_to_remove:
            del self._circuit_breakers[domain]
            removed_count += 1

        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} idle circuit breakers")

        return removed_count
