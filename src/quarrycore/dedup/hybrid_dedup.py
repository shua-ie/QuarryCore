"""
Hybrid Deduplication Orchestrator.

Combines exact hash and near-duplicate detection layers:
1. Exact layer: Canonical HTML → SHA-256 → SQLite WAL storage
2. Near layer: MinHashLSH → Redis with fakeredis fallback

Features:
- Fast-return on exact hits (short-circuit evaluation)
- Prometheus metrics for monitoring
- Resilient operation (Redis down = disable near-dup layer)
- Simple is_duplicate(doc: ExtractResult) -> bool API
- Legacy adapter for backward compatibility
- Structured logging with performance tracking
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import structlog
from prometheus_client import Counter, Histogram

from .canonical import CanonicalHTMLProcessor
from .hash_db import HashDatabase
from .minhash_redis import RedisMinHashLSH

logger = structlog.get_logger(__name__)

# Prometheus metrics
dedup_exact_hits_total = Counter("dedup_exact_hits_total", "Total number of exact duplicate hits")
dedup_near_hits_total = Counter("dedup_near_hits_total", "Total number of near duplicate hits")
dedup_latency_seconds = Histogram(
    "dedup_latency_seconds",
    "Deduplication check latency in seconds",
    ["layer"],
    buckets=[0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0],
)


@dataclass
class DedupConfig:
    """Configuration for hybrid deduplication."""

    sqlite_path: str = "data/dedup.db"
    redis_url: str = "redis://localhost:6379/0"
    minhash_shingle_size: int = 7
    minhash_num_perm: int = 128
    minhash_threshold: float = 0.85
    minhash_enabled: bool = True


class ExtractResult:
    """Minimal ExtractResult for compatibility."""

    def __init__(self, text: str = "", html: str = "", url: str = "unknown"):
        self.text = text
        self.html = html
        self.url = url


class HybridDeduplicator:
    """
    Production-grade two-layer deduplication system.

    Architecture:
    1. Canonical HTML → Exact SHA-256 hash → SQLite WAL storage
    2. Text content → MinHashLSH → Redis backend

    Features:
    - Fast exact duplicate detection with immediate return
    - Near-duplicate detection with resilient Redis backend
    - Prometheus metrics and structured logging
    - Graceful degradation when components fail
    - Simple boolean API with legacy adapter
    """

    def __init__(self, config: DedupConfig):
        """
        Initialize hybrid deduplicator.

        Args:
            config: Deduplication configuration
        """
        self.config = config

        # Initialize components
        self._init_components()

        # Performance tracking
        self._total_checks = 0
        self._exact_hits = 0
        self._near_hits = 0
        self._start_time = time.time()

        logger.info(
            "Initialized HybridDeduplicator",
            sqlite_path=self.config.sqlite_path,
            redis_url=self.config.redis_url,
            minhash_enabled=self.config.minhash_enabled,
        )

    def _init_components(self) -> None:
        """Initialize deduplication components."""
        try:
            # HTML canonicalization processor
            self.canonical = CanonicalHTMLProcessor()

            # Exact hash database (SQLite WAL)
            self.hash_db = HashDatabase(db_path=Path(self.config.sqlite_path), wal_mode=True)

            # Near-duplicate detection (Redis MinHashLSH)
            if self.config.minhash_enabled:
                self.minhash = RedisMinHashLSH(
                    redis_url=self.config.redis_url,
                    threshold=self.config.minhash_threshold,
                    num_perm=self.config.minhash_num_perm,
                    shingle_size=self.config.minhash_shingle_size,
                    enabled=True,
                )
            else:
                self.minhash = None

            logger.info("All deduplication components initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize deduplication components: {e}")
            raise

    async def is_duplicate(self, doc: ExtractResult) -> bool:
        """
        Check if document is a duplicate (simple boolean API).

        Args:
            doc: Extracted document with text/html content

        Returns:
            True if duplicate, False if unique
        """
        start_time = time.time()
        self._total_checks += 1

        try:
            # Stage 1: Exact duplicate check using canonical HTML
            is_exact_dup = await self._check_exact_duplicate(doc)

            if is_exact_dup:
                # Fast return on exact hit
                self._exact_hits += 1
                dedup_exact_hits_total.inc()

                with dedup_latency_seconds.labels(layer="exact").time():
                    pass  # Already timed above

                logger.debug("Exact duplicate detected", url=doc.url, duration_ms=(time.time() - start_time) * 1000)
                return True

            # Stage 2: Near-duplicate check (if enabled and Redis available)
            if self.minhash and self.minhash.enabled:
                is_near_dup = await self._check_near_duplicate(doc)

                if is_near_dup:
                    self._near_hits += 1
                    dedup_near_hits_total.inc()

                    with dedup_latency_seconds.labels(layer="near").time():
                        pass

                    logger.debug("Near duplicate detected", url=doc.url, duration_ms=(time.time() - start_time) * 1000)
                    return True

            # Document is unique
            logger.debug("Document is unique", url=doc.url, duration_ms=(time.time() - start_time) * 1000)
            return False

        except Exception as e:
            logger.error(
                "Deduplication check failed", url=doc.url, error=str(e), duration_ms=(time.time() - start_time) * 1000
            )
            # On error, assume unique to avoid blocking pipeline
            return False

        finally:
            # Record overall latency
            duration = time.time() - start_time
            dedup_latency_seconds.labels(layer="total").observe(duration)

    async def _check_exact_duplicate(self, doc: ExtractResult) -> bool:
        """Check for exact duplicates using canonical HTML hash."""
        try:
            # Use HTML if available, fallback to text
            content = doc.html if doc.html else doc.text

            if not content:
                return False

            # Canonicalize HTML for consistent hashing
            canonical_content = self.canonical.canonicalize(content)

            # Check against hash database
            is_dup, content_hash = await self.hash_db.is_duplicate(canonical_content, doc.url)

            return is_dup

        except Exception as e:
            logger.warning(f"Exact duplicate check failed: {e}")
            return False

    async def _check_near_duplicate(self, doc: ExtractResult) -> bool:
        """Check for near duplicates using MinHashLSH."""
        try:
            if not self.minhash or not doc.text:
                return False

            # Use URL as document ID (could be improved with better ID strategy)
            doc_id = doc.url

            # Check for similar documents
            similar_docs = await self.minhash.is_near_duplicate(doc.text, doc_id)

            return len(similar_docs) > 0

        except Exception as e:
            logger.warning(f"Near duplicate check failed: {e}")
            return False

    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive deduplication statistics."""
        uptime = time.time() - self._start_time

        stats = {
            "total_checks": self._total_checks,
            "exact_hits": self._exact_hits,
            "near_hits": self._near_hits,
            "unique_docs": self._total_checks - self._exact_hits - self._near_hits,
            "exact_hit_rate": self._exact_hits / max(1, self._total_checks),
            "near_hit_rate": self._near_hits / max(1, self._total_checks),
            "overall_duplicate_rate": (self._exact_hits + self._near_hits) / max(1, self._total_checks),
            "uptime_seconds": uptime,
            "checks_per_second": self._total_checks / max(1, uptime),
        }

        # Add component stats
        try:
            stats["canonical_stats"] = self.canonical.get_stats()
            stats["hash_db_stats"] = await self.hash_db.get_stats()

            if self.minhash:
                stats["minhash_stats"] = self.minhash.get_stats()
        except Exception as e:
            logger.warning(f"Error collecting component stats: {e}")

        return stats

    async def close(self) -> None:
        """Close all connections and cleanup resources."""
        try:
            await self.hash_db.close()

            if self.minhash:
                await self.minhash.close()

            logger.info("HybridDeduplicator closed successfully")

        except Exception as e:
            logger.error(f"Error closing HybridDeduplicator: {e}")


# Legacy adapter for backward compatibility
class LegacyDuplicationResult:
    """
    Legacy adapter for existing DuplicationResult API.

    Provides rich metadata while maintaining backward compatibility.
    """

    def __init__(self, is_duplicate: bool, content_hash: str = "", duplicate_type: str = ""):
        self.is_duplicate = is_duplicate
        self.content_hash = content_hash
        self.duplicate_type = duplicate_type
        self.confidence_score = 1.0 if is_duplicate else 0.0
        self.processing_time_ms = 0.0


async def check_duplicates_legacy(
    deduplicator: HybridDeduplicator, content: Any, metadata: Any
) -> LegacyDuplicationResult:
    """
    Legacy adapter function for existing deduplication API.

    Args:
        deduplicator: HybridDeduplicator instance
        content: ExtractedContent object
        metadata: ContentMetadata object

    Returns:
        LegacyDuplicationResult with rich metadata
    """
    start_time = time.time()

    try:
        # Convert to new ExtractResult format
        doc = ExtractResult(
            text=getattr(content, "text", ""),
            html=getattr(content, "html", ""),
            url=getattr(metadata, "url", "unknown"),
        )

        # Perform deduplication check
        is_dup = await deduplicator.is_duplicate(doc)

        # Create legacy result
        result = LegacyDuplicationResult(
            is_duplicate=is_dup,
            content_hash="",
            duplicate_type="exact" if is_dup else "",  # Not exposed in simple API
        )

        result.processing_time_ms = (time.time() - start_time) * 1000
        return result

    except Exception as e:
        logger.error(f"Legacy deduplication check failed: {e}")
        result = LegacyDuplicationResult(is_duplicate=False)
        result.processing_time_ms = (time.time() - start_time) * 1000
        return result
