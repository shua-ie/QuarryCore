"""
Production-Grade Hybrid Deduplication System for QuarryCore.

Two-layer deduplication:
1. Exact layer: Canonical HTML → SHA-256 → SQLite WAL storage
2. Near layer: MinHashLSH (datasketch) → Redis backend

Key features:
- Canonical HTML normalization (script/style removal, whitespace collapse)
- SQLite WAL mode with unique hash index
- Redis-backed LSH with fallback to fakeredis
- Prometheus metrics (dedup_exact_hits_total, dedup_near_hits_total, dedup_latency_seconds)
- Resilient operation (Redis down = disable near-dup layer)
- Simple is_duplicate(doc: ExtractResult) -> bool API
"""

from .canonical import CanonicalHTMLProcessor
from .hash_db import HashDatabase
from .hybrid_dedup import HybridDeduplicator
from .minhash_redis import RedisMinHashLSH

__all__ = [
    "CanonicalHTMLProcessor",
    "HashDatabase",
    "RedisMinHashLSH",
    "HybridDeduplicator",
]
