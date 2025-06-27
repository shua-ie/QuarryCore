"""
Enhanced error recovery system for QuarryCore.

Provides dead letter queues and persistent circuit breakers for fault tolerance.
"""

from .circuit_breaker import CircuitBreakerState, PersistentCircuitBreaker
from .dead_letter import DeadLetterQueue, FailedDocument

__all__ = [
    "DeadLetterQueue",
    "FailedDocument",
    "PersistentCircuitBreaker",
    "CircuitBreakerState",
]
