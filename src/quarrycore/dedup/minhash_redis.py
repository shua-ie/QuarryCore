"""
Redis-backed MinHashLSH for Near-Duplicate Detection.

Implements near-duplicate detection using:
- datasketch.MinHashLSH with Redis backend
- 7-character shingles for text similarity
- Threshold=0.85, num_perm=128 for optimal production balance
- Resilient Redis connection with fallback to fakeredis
- Lazy connection and graceful degradation when Redis unavailable
"""

import hashlib
import re
import time
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urlparse

try:
    import redis

    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False

try:
    import fakeredis

    HAS_FAKEREDIS = True
except ImportError:
    HAS_FAKEREDIS = False

try:
    from datasketch import MinHash, MinHashLSH

    HAS_DATASKETCH = True
except ImportError:
    HAS_DATASKETCH = False

import structlog

logger = structlog.get_logger(__name__)


class RedisMinHashLSH:
    """
    Redis-backed MinHash LSH for near-duplicate detection.

    Features:
    - Uses datasketch.MinHashLSH with Redis storage
    - 7-character shingles for optimal text similarity detection
    - Threshold=0.85, num_perm=128 for production balance
    - Lazy Redis connection with automatic fallback
    - Graceful degradation when Redis unavailable
    - Metrics tracking for monitoring
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        threshold: float = 0.85,
        num_perm: int = 128,
        shingle_size: int = 7,
        enabled: bool = True,
    ):
        """
        Initialize Redis MinHash LSH.

        Args:
            redis_url: Redis connection URL
            threshold: Jaccard similarity threshold (0.85 per spec)
            num_perm: Number of permutations (128 per spec)
            shingle_size: Character shingle size (7 per spec)
            enabled: Enable near-duplicate detection
        """
        self.redis_url = redis_url
        self.threshold = threshold
        self.num_perm = num_perm
        self.shingle_size = shingle_size
        self.enabled = enabled

        # Dependencies check
        if not HAS_DATASKETCH:
            logger.error("datasketch not available - near-dup detection disabled")
            self.enabled = False
            return

        # Connection state
        self._redis_client: Optional[Any] = None
        self._lsh_index: Optional[MinHashLSH] = None
        self._connection_failed = False
        self._last_connection_attempt = 0
        self._connection_retry_delay = 60  # seconds

        # Statistics
        self._total_checks = 0
        self._near_duplicate_hits = 0
        self._redis_errors = 0
        self._fallback_used = False

        # Performance tracking
        self._last_error_log = 0
        self._error_log_interval = 60  # Log Redis errors max once per minute

        # Initialize connection
        if self.enabled:
            self._init_connection()

    def _init_connection(self) -> None:
        """Initialize Redis connection and LSH index."""
        try:
            # Try Redis connection
            if HAS_REDIS:
                self._redis_client = redis.from_url(
                    self.redis_url,
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=5,
                    retry_on_timeout=True,
                    health_check_interval=30,
                )

                # Test connection
                self._redis_client.ping()
                logger.info(f"Connected to Redis at {self.redis_url}")

            else:
                raise ImportError("redis package not available")

        except Exception as e:
            # Fallback to fakeredis if available
            if HAS_FAKEREDIS:
                logger.warning(f"Redis connection failed ({e}), using fakeredis fallback")
                self._redis_client = fakeredis.FakeRedis(decode_responses=True)
                self._fallback_used = True
            else:
                logger.error(f"Redis and fakeredis unavailable - near-dup disabled: {e}")
                self.enabled = False
                return

        try:
            # Initialize MinHashLSH with Redis backend
            self._lsh_index = MinHashLSH(
                threshold=self.threshold,
                num_perm=self.num_perm,
                storage_config={"type": "redis", "redis": {"host": self._redis_client}},
            )

            logger.info(
                f"Initialized MinHashLSH: threshold={self.threshold}, "
                f"num_perm={self.num_perm}, shingle_size={self.shingle_size}"
            )

        except Exception as e:
            logger.error(f"Failed to initialize MinHashLSH: {e}")
            self.enabled = False

    def _should_retry_connection(self) -> bool:
        """Check if we should retry Redis connection."""
        current_time = time.time()
        return current_time - self._last_connection_attempt > self._connection_retry_delay

    def _create_shingles(self, text: str) -> Set[str]:
        """
        Create character shingles from text.

        Args:
            text: Input text

        Returns:
            Set of character shingles
        """
        if not text or len(text) < self.shingle_size:
            return {text} if text else set()

        # Normalize text: lowercase, remove extra whitespace
        normalized = re.sub(r"\s+", " ", text.lower().strip())

        # Create character shingles
        shingles = set()
        for i in range(len(normalized) - self.shingle_size + 1):
            shingle = normalized[i : i + self.shingle_size]
            shingles.add(shingle)

        return shingles

    def _create_minhash(self, text: str) -> MinHash:
        """
        Create MinHash signature from text.

        Args:
            text: Input text

        Returns:
            MinHash signature
        """
        minhash = MinHash(num_perm=self.num_perm)
        shingles = self._create_shingles(text)

        for shingle in shingles:
            minhash.update(shingle.encode("utf-8"))

        return minhash

    async def is_near_duplicate(self, text: str, doc_id: str) -> List[str]:
        """
        Check if text is a near-duplicate and add to index.

        Args:
            text: Text content to check
            doc_id: Document identifier

        Returns:
            List of similar document IDs (empty if unique)
        """
        if not self.enabled:
            return []

        self._total_checks += 1

        # Retry connection if needed
        if self._connection_failed and self._should_retry_connection():
            logger.info("Retrying Redis connection for MinHashLSH")
            self._last_connection_attempt = time.time()
            self._init_connection()
            if not self.enabled:
                return []

        try:
            # Create MinHash signature
            minhash = self._create_minhash(text)

            # Query for similar documents
            similar_docs = list(self._lsh_index.query(minhash))

            # Add current document to index
            self._lsh_index.insert(doc_id, minhash)

            if similar_docs:
                self._near_duplicate_hits += 1
                logger.debug(f"Near-duplicate found: {doc_id} similar to {similar_docs}")

            return similar_docs

        except Exception as e:
            self._redis_errors += 1
            self._connection_failed = True

            # Log errors at most once per minute to avoid spam
            current_time = time.time()
            if current_time - self._last_error_log > self._error_log_interval:
                logger.warning(f"MinHashLSH operation failed: {e}")
                self._last_error_log = current_time

            # Return empty list (treat as unique) to avoid blocking pipeline
            return []

    async def batch_check(self, documents: List[tuple[str, str]]) -> List[List[str]]:
        """
        Batch check multiple documents for near-duplicates.

        Args:
            documents: List of (text, doc_id) tuples

        Returns:
            List of similar document lists for each input
        """
        if not self.enabled:
            return [[] for _ in documents]

        results = []

        for text, doc_id in documents:
            similar_docs = await self.is_near_duplicate(text, doc_id)
            results.append(similar_docs)

        return results

    async def get_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate Jaccard similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Jaccard similarity score (0.0 to 1.0)
        """
        if not self.enabled:
            return 0.0

        try:
            minhash1 = self._create_minhash(text1)
            minhash2 = self._create_minhash(text2)

            return minhash1.jaccard(minhash2)

        except Exception as e:
            logger.warning(f"Similarity calculation failed: {e}")
            return 0.0

    async def clear_index(self) -> bool:
        """
        Clear all data from the LSH index.

        Returns:
            True if successful, False otherwise
        """
        if not self.enabled or not self._redis_client:
            return False

        try:
            # Clear Redis keys used by MinHashLSH
            keys = self._redis_client.keys("lsh:*")
            if keys:
                self._redis_client.delete(*keys)
                logger.info(f"Cleared {len(keys)} LSH keys from Redis")

            # Reinitialize LSH index
            self._init_connection()
            return True

        except Exception as e:
            logger.error(f"Failed to clear LSH index: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get MinHashLSH statistics."""
        stats = {
            "enabled": self.enabled,
            "redis_url": self.redis_url,
            "threshold": self.threshold,
            "num_perm": self.num_perm,
            "shingle_size": self.shingle_size,
            "total_checks": self._total_checks,
            "near_duplicate_hits": self._near_duplicate_hits,
            "redis_errors": self._redis_errors,
            "connection_failed": self._connection_failed,
            "fallback_used": self._fallback_used,
            "duplicate_rate": self._near_duplicate_hits / max(1, self._total_checks),
        }

        # Add Redis info if available
        if self._redis_client and not self._connection_failed:
            try:
                info = self._redis_client.info()
                stats["redis_memory_used"] = info.get("used_memory", 0)
                stats["redis_connected_clients"] = info.get("connected_clients", 0)

                # Count LSH-related keys
                lsh_keys = self._redis_client.keys("lsh:*")
                stats["lsh_keys_count"] = len(lsh_keys)

            except Exception as e:
                logger.debug(f"Could not get Redis stats: {e}")

        return stats

    async def close(self) -> None:
        """Close Redis connection and cleanup."""
        if self._redis_client:
            try:
                if hasattr(self._redis_client, "close"):
                    self._redis_client.close()
                logger.info("Redis connection closed")
            except Exception as e:
                logger.warning(f"Error closing Redis connection: {e}")

        self._redis_client = None
        self._lsh_index = None
