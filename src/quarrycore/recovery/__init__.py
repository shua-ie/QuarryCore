"""
Enhanced error recovery system for QuarryCore.

Provides dead letter queues and persistent circuit breakers for fault tolerance.
"""

from .dead_letter import DeadLetterQueue, FailedDocument
from .circuit_breaker import PersistentCircuitBreaker, CircuitBreakerState

__all__ = [
    "DeadLetterQueue",
    "FailedDocument",
    "PersistentCircuitBreaker",
    "CircuitBreakerState",
] 