"""
QuarryCore Deduplication System.

A comprehensive 4-level deduplication system for AI training data:
- Level 1: SHA-256 exact hash with bloom filter
- Level 2: MinHash LSH for near-duplicates
- Level 3: GPU-accelerated semantic similarity
- Level 4: Fuzzy matching for partial overlaps
"""

from .bloom_filter import BloomFilterConfig, ShardedBloomFilter
from .deduplicator import DeduplicationConfig, DuplicateType, MultiLevelDeduplicator
from .fuzzy_matcher import FuzzyConfig, FuzzyMatcher
from .minhash_lsh import MinHashConfig, MinHashLSHDeduplicator
from .semantic_dedup import SemanticConfig, SemanticDeduplicator

__all__ = [
    # Main deduplicator
    "MultiLevelDeduplicator",
    "DeduplicationConfig",
    "DuplicateType",
    # Level 1: Bloom Filter
    "ShardedBloomFilter",
    "BloomFilterConfig",
    # Level 2: MinHash
    "MinHashLSHDeduplicator",
    "MinHashConfig",
    # Level 3: Semantic
    "SemanticDeduplicator",
    "SemanticConfig",
    # Level 4: Fuzzy
    "FuzzyMatcher",
    "FuzzyConfig",
]
