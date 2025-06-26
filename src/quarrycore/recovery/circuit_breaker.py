"""
Persistent circuit breaker for QuarryCore.

Provides circuit breaker pattern with state persistence across application restarts.
"""
from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Awaitable
from uuid import uuid4

import aiofiles


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


class PersistentCircuitBreaker:
    """
    Circuit breaker with persistent state across restarts.
    
    Features:
    - State persistence across application restarts
    - Configurable failure thresholds
    - Health check integration
    - Automatic recovery testing
    - Per-domain or per-service isolation
    """
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        success_threshold: int = 3,
        timeout_duration: float = 60.0,
        half_open_max_calls: int = 3,
        state_file: Optional[Path] = None,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.timeout_duration = timeout_duration
        self.half_open_max_calls = half_open_max_calls
        
        # State file for persistence
        if state_file:
            self.state_file = state_file
        else:
            self.state_file = Path(f"./data/circuit_breakers/{name}.json")
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Circuit breaker state
        self._state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._half_open_calls = 0
        self._last_health_check: Optional[float] = None
        
        # Lock for thread safety
        self._lock = asyncio.Lock()
        
        # Load persisted state
        asyncio.create_task(self._load_state())
    
    async def can_execute(self) -> bool:
        """Check if the circuit breaker allows execution."""
        async with self._lock:
            if self._state == CircuitBreakerState.CLOSED:
                return True
            
            if self._state == CircuitBreakerState.OPEN:
                # Check if timeout has passed
                if self._last_failure_time and \
                   time.time() - self._last_failure_time >= self.timeout_duration:
                    # Transition to half-open
                    await self._transition_to_half_open()
                    return True
                return False
            
            if self._state == CircuitBreakerState.HALF_OPEN:
                # Allow limited calls in half-open state
                if self._half_open_calls < self.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False
            
            return False
    
    async def record_success(self) -> None:
        """Record a successful execution."""
        async with self._lock:
            if self._state == CircuitBreakerState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.success_threshold:
                    await self._transition_to_closed()
            elif self._state == CircuitBreakerState.CLOSED:
                # Reset failure count on success
                self._failure_count = 0
            
            await self._save_state()
    
    async def record_failure(self) -> None:
        """Record a failed execution."""
        async with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            
            if self._state == CircuitBreakerState.CLOSED:
                if self._failure_count >= self.failure_threshold:
                    await self._transition_to_open()
            elif self._state == CircuitBreakerState.HALF_OPEN:
                # Any failure in half-open goes back to open
                await self._transition_to_open()
            
            await self._save_state()
    
    async def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state."""
        async with self._lock:
            return {
                "name": self.name,
                "state": self._state.value,
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "last_failure_time": self._last_failure_time,
                "half_open_calls": self._half_open_calls,
                "time_until_retry": self._get_time_until_retry(),
            }
    
    async def reset(self) -> None:
        """Manually reset the circuit breaker."""
        async with self._lock:
            self._state = CircuitBreakerState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._last_failure_time = None
            self._half_open_calls = 0
            await self._save_state()
    
    async def health_check(self, check_function: Callable[[], Awaitable[bool]]) -> bool:
        """
        Perform a health check to test recovery.
        
        Args:
            check_function: Async function that returns True if healthy
            
        Returns:
            True if health check passed
        """
        # Don't check too frequently
        if self._last_health_check and \
           time.time() - self._last_health_check < 30:
            return False
        
        self._last_health_check = time.time()
        
        try:
            is_healthy = await asyncio.wait_for(
                check_function(),
                timeout=5.0
            )
            
            if is_healthy and self._state == CircuitBreakerState.OPEN:
                async with self._lock:
                    await self._transition_to_half_open()
            
            return is_healthy
            
        except asyncio.TimeoutError:
            return False
        except Exception:
            return False
    
    def _get_time_until_retry(self) -> float:
        """Get seconds until circuit breaker allows retry."""
        if self._state != CircuitBreakerState.OPEN or not self._last_failure_time:
            return 0.0
        
        elapsed = time.time() - self._last_failure_time
        remaining = self.timeout_duration - elapsed
        return max(0.0, remaining)
    
    async def _transition_to_open(self) -> None:
        """Transition to open state."""
        self._state = CircuitBreakerState.OPEN
        self._failure_count = 0
        self._success_count = 0
        self._half_open_calls = 0
    
    async def _transition_to_half_open(self) -> None:
        """Transition to half-open state."""
        self._state = CircuitBreakerState.HALF_OPEN
        self._failure_count = 0
        self._success_count = 0
        self._half_open_calls = 0
    
    async def _transition_to_closed(self) -> None:
        """Transition to closed state."""
        self._state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._half_open_calls = 0
        self._last_failure_time = None
    
    async def _save_state(self) -> None:
        """Save circuit breaker state to file."""
        state_data = {
            "name": self.name,
            "state": self._state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "last_failure_time": self._last_failure_time,
            "half_open_calls": self._half_open_calls,
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        try:
            async with aiofiles.open(self.state_file, 'w') as f:
                await f.write(json.dumps(state_data, indent=2))
        except Exception as e:
            # Log error but don't fail
            print(f"Failed to save circuit breaker state: {e}")
    
    async def _load_state(self) -> None:
        """Load circuit breaker state from file."""
        if not self.state_file.exists():
            return
        
        try:
            async with aiofiles.open(self.state_file, 'r') as f:
                content = await f.read()
                state_data = json.loads(content)
            
            # Restore state
            self._state = CircuitBreakerState(state_data["state"])
            self._failure_count = state_data.get("failure_count", 0)
            self._success_count = state_data.get("success_count", 0)
            self._last_failure_time = state_data.get("last_failure_time")
            self._half_open_calls = state_data.get("half_open_calls", 0)
            
            # Check if we should transition based on timeout
            if self._state == CircuitBreakerState.OPEN and self._last_failure_time:
                if time.time() - self._last_failure_time >= self.timeout_duration:
                    await self._transition_to_half_open()
                    await self._save_state()
        
        except Exception as e:
            # Log error but continue with default state
            print(f"Failed to load circuit breaker state: {e}")


class CircuitBreakerManager:
    """Manages multiple circuit breakers."""
    
    def __init__(self, state_dir: Path = Path("./data/circuit_breakers")):
        self.state_dir = state_dir
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._breakers: Dict[str, PersistentCircuitBreaker] = {}
    
    def get_breaker(
        self,
        name: str,
        failure_threshold: int = 5,
        timeout_duration: float = 60.0,
    ) -> PersistentCircuitBreaker:
        """Get or create a circuit breaker."""
        if name not in self._breakers:
            self._breakers[name] = PersistentCircuitBreaker(
                name=name,
                failure_threshold=failure_threshold,
                timeout_duration=timeout_duration,
                state_file=self.state_dir / f"{name}.json",
            )
        
        return self._breakers[name]
    
    async def get_all_states(self) -> Dict[str, Dict[str, Any]]:
        """Get states of all circuit breakers."""
        states = {}
        for name, breaker in self._breakers.items():
            states[name] = await breaker.get_state()
        return states
    
    async def reset_all(self) -> None:
        """Reset all circuit breakers."""
        for breaker in self._breakers.values():
            await breaker.reset() 