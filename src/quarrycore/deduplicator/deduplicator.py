"""
Multi-level deduplication orchestrator.

Combines all 4 levels of deduplication:
1. SHA-256 exact hash with bloom filter
2. MinHash LSH for near-duplicates
3. GPU-accelerated semantic embeddings
4. Fuzzy matching for partial overlaps
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple
from uuid import UUID

# NumPy imports with graceful fallbacks (proven pattern)
try:
    import numpy as np  # type: ignore[import-not-found]

    HAS_NUMPY = True
except ImportError:
    if TYPE_CHECKING:
        import numpy as np  # type: ignore[import-not-found]
    else:
        np = None
    HAS_NUMPY = False

from ..protocols import ContentMetadata, DeduplicatorProtocol, DuplicationResult, ExtractedContent, HardwareCapabilities
from .bloom_filter import BloomFilterConfig, ShardedBloomFilter
from .fuzzy_matcher import FuzzyConfig, FuzzyMatcher
from .minhash_lsh import MinHashConfig, MinHashLSHDeduplicator
from .semantic_dedup import SemanticConfig, SemanticDeduplicator

logger = logging.getLogger(__name__)


class DuplicateType(Enum):
    """Types of duplicates detected."""

    EXACT = "exact"
    NEAR_DUPLICATE = "near_duplicate"
    SEMANTIC_SIMILAR = "semantic_similar"
    FUZZY_MATCH = "fuzzy_match"
    PARTIAL_OVERLAP = "partial_overlap"


@dataclass
class DeduplicationConfig:
    """Configuration for multi-level deduplication."""

    # Level 1: Bloom Filter
    bloom_capacity: int = 10_000_000
    bloom_error_rate: float = 0.001

    # Level 2: MinHash LSH
    minhash_permutations: int = 128
    minhash_threshold: float = 0.8

    # Level 3: Semantic
    semantic_model: str = "all-MiniLM-L6-v2"
    semantic_batch_size: int = 64
    semantic_threshold: float = 0.85
    use_gpu: bool = True

    # Level 4: Fuzzy
    fuzzy_algorithm: str = "token_sort_ratio"
    fuzzy_threshold: float = 0.85

    # General
    storage_path: Path = Path("data/deduplication")
    enable_all_levels: bool = True
    parallel_processing: bool = True

    # Content type specific thresholds
    content_type_configs: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.content_type_configs:
            self.content_type_configs = {
                "medical": {"minhash": 0.85, "semantic": 0.9, "fuzzy": 0.92},
                "legal": {"minhash": 0.83, "semantic": 0.88, "fuzzy": 0.9},
                "technical": {"minhash": 0.8, "semantic": 0.85, "fuzzy": 0.87},
                "ecommerce": {"minhash": 0.75, "semantic": 0.8, "fuzzy": 0.83},
                "general": {"minhash": 0.8, "semantic": 0.85, "fuzzy": 0.85},
            }


class MultiLevelDeduplicator(DeduplicatorProtocol):
    """
    Production-grade 4-level deduplication system.

    Features:
    - Level 1: SHA-256 exact matching with sharded bloom filters
    - Level 2: MinHash LSH for near-duplicate detection
    - Level 3: GPU-accelerated semantic similarity with FAISS
    - Level 4: Fuzzy matching for partial overlaps
    - Adaptive hardware optimization
    - Incremental index updates
    - Persistent storage
    """

    def __init__(
        self,
        config: DeduplicationConfig,
        hardware: Optional[HardwareCapabilities] = None,
    ) -> None:
        self.config = config
        self.hardware = hardware or HardwareCapabilities()

        # Create storage directories
        self.config.storage_path.mkdir(parents=True, exist_ok=True)

        # Initialize components based on hardware
        self._init_components()

        # Statistics
        self.stats: Dict[str, Any] = {
            "total_processed": 0,
            "exact_duplicates": 0,
            "near_duplicates": 0,
            "semantic_duplicates": 0,
            "fuzzy_matches": 0,
            "unique_documents": 0,
            "processing_time": 0.0,
        }

        logger.info(
            f"Initialized MultiLevelDeduplicator with {4 if self.config.enable_all_levels else 1} levels, "
            f"GPU: {self.config.use_gpu and self.hardware.has_gpu}"
        )

    def _init_components(self) -> None:
        """Initialize deduplication components based on hardware."""
        # Level 1: Bloom Filter (always enabled)
        bloom_config = BloomFilterConfig(
            capacity=self.config.bloom_capacity,
            error_rate=self.config.bloom_error_rate,
            shard_dir=self.config.storage_path / "bloom",
            num_shards=min(16, self.hardware.cpu_cores * 2) if self.hardware else 4,
        )
        self.bloom_filter = ShardedBloomFilter(bloom_config, self.hardware)

        if not self.config.enable_all_levels:
            self.minhash: Optional[MinHashLSHDeduplicator] = None
            self.semantic: Optional[SemanticDeduplicator] = None
            self.fuzzy: Optional[FuzzyMatcher] = None
            return

        # Level 2: MinHash LSH
        minhash_config = MinHashConfig(
            num_perm=self.config.minhash_permutations,
            threshold=self.config.minhash_threshold,
            storage_path=self.config.storage_path / "minhash",
        )
        self.minhash = MinHashLSHDeduplicator(minhash_config, self.hardware)

        # Level 3: Semantic (with GPU support)
        semantic_config = SemanticConfig(
            model_name=self.config.semantic_model,
            batch_size=self.config.semantic_batch_size,
            similarity_threshold=self.config.semantic_threshold,
            use_gpu=self.config.use_gpu and self.hardware.has_gpu,
            storage_path=self.config.storage_path / "semantic",
            content_type_thresholds=self._get_semantic_thresholds(),
        )

        try:
            self.semantic = SemanticDeduplicator(semantic_config, self.hardware)
        except ImportError as e:
            logger.warning(f"Semantic deduplication unavailable: {e}")
            self.semantic = None

        # Level 4: Fuzzy Matching
        fuzzy_config = FuzzyConfig(
            algorithm=self.config.fuzzy_algorithm,
            min_similarity=self.config.fuzzy_threshold,
            storage_path=self.config.storage_path / "fuzzy",
            content_type_thresholds=self._get_fuzzy_thresholds(),
        )
        self.fuzzy = FuzzyMatcher(fuzzy_config, self.hardware)

    def _get_semantic_thresholds(self) -> Dict[str, float]:
        """Get semantic thresholds per content type."""
        return {
            content_type: config.get("semantic", self.config.semantic_threshold)
            for content_type, config in self.config.content_type_configs.items()
        }

    def _get_fuzzy_thresholds(self) -> Dict[str, float]:
        """Get fuzzy thresholds per content type."""
        return {
            content_type: config.get("fuzzy", self.config.fuzzy_threshold)
            for content_type, config in self.config.content_type_configs.items()
        }

    async def check_duplicates(
        self,
        content: ExtractedContent,
        metadata: ContentMetadata,
        *,
        enable_semantic: bool = True,
        hardware_caps: Optional[HardwareCapabilities] = None,
    ) -> DuplicationResult:
        """
        Check if content is duplicate using all 4 levels.
        """
        start_time = time.time()

        # Determine content type and get identifier
        content_type = self._determine_content_type(metadata)
        text = content.text
        doc_id = metadata.url  # Use URL from metadata as the document ID

        # Level 1: Exact duplicate check
        is_new, content_hash = self.bloom_filter.add(text)

        # Create UUID from first 32 chars of SHA-256 hash
        content_id = UUID(hex=content_hash[:32])
        result = DuplicationResult(content_hash=content_hash, content_id=content_id)

        if not is_new:
            self.stats["exact_duplicates"] += 1
            result.is_duplicate = True
            result.exact_match = True
            result.duplicate_type = DuplicateType.EXACT.value
            result.confidence_score = 1.0
        else:
            # Continue with other levels if enabled
            print(f"DEBUG: is_new=True, enable_all_levels={self.config.enable_all_levels}")
            if self.config.enable_all_levels:
                print(f"DEBUG: Calling _check_advanced_levels for {doc_id}")
                advanced_result = await self._check_advanced_levels(doc_id, text, content_type)
                # Merge results
                if advanced_result.is_duplicate:
                    result = advanced_result
                    result.content_hash = content_hash
                    result.content_id = UUID(hex=content_hash[:32])

        self.stats["total_processed"] += 1
        result.processing_time_ms = (time.time() - start_time) * 1000

        if not result.is_duplicate:
            self.stats["unique_documents"] += 1

        return result

    async def _check_advanced_levels(self, doc_id: str, text: str, content_type: str) -> DuplicationResult:
        """Check levels 2-4 for duplicates."""
        print(f"DEBUG: In _check_advanced_levels for {doc_id}")
        print(f"DEBUG: MinHash enabled: {self.minhash is not None}")
        print(f"DEBUG: Fuzzy enabled: {self.fuzzy is not None}")

        if not self.config.enable_all_levels:
            return DuplicationResult(
                is_duplicate=False,
                duplicate_type="",
            )

        # Prepare tasks for parallel execution
        tasks: List[Any] = []

        print("DEBUG: About to check components")
        print(f"DEBUG: self.minhash = {self.minhash}")
        print(f"DEBUG: self.semantic = {self.semantic}")
        print(f"DEBUG: self.fuzzy = {self.fuzzy}")

        # Level 2: MinHash LSH
        if self.minhash is not None:
            tasks.append(self._check_minhash(doc_id, text))

        # Level 3: Semantic similarity
        if self.semantic is not None:
            tasks.append(self._check_semantic(doc_id, text, content_type))

        # Level 4: Fuzzy matching
        if self.fuzzy is not None:
            tasks.append(self._check_fuzzy(doc_id, text, content_type))

        print(f"DEBUG: Total tasks created: {len(tasks)}")

        # Execute checks in parallel if configured
        if self.config.parallel_processing and tasks:
            print(f"DEBUG: Executing {len(tasks)} tasks in parallel")
            results = await asyncio.gather(*tasks)
        else:
            print(f"DEBUG: Executing {len(tasks)} tasks sequentially")
            results = []
            for task in tasks:
                result = await task
                results.append(result)

        # Combine results
        return self._combine_results(results)

    async def _check_minhash(self, doc_id: str, text: str) -> DuplicationResult:
        """Check for near-duplicates using MinHash."""
        logger.debug(f"_check_minhash called for doc_id: {doc_id}")
        result = DuplicationResult()
        if self.minhash is not None:
            similar_docs = await asyncio.to_thread(self.minhash.add, doc_id, text)
            logger.debug(f"MinHash found {len(similar_docs)} similar docs")

            if similar_docs:
                self.stats["near_duplicates"] += 1
                result.is_duplicate = True
                result.duplicate_type = DuplicateType.NEAR_DUPLICATE.value
                result.jaccard_similarity = self.config.minhash_threshold
                # Convert doc IDs to valid UUIDs
                result.near_duplicates = []
                for doc in similar_docs:
                    try:
                        result.near_duplicates.append(UUID(doc))
                    except ValueError:
                        # If not a valid UUID, generate one from the doc ID using secure hash
                        import hashlib

                        doc_hash = hashlib.sha256(doc.encode()).hexdigest()
                        result.near_duplicates.append(UUID(hex=doc_hash[:32]))
                result.confidence_score = self.config.minhash_threshold
        return result

    async def _check_semantic(self, doc_id: str, text: str, content_type: str) -> DuplicationResult:
        """Check for semantic similarity."""
        result = DuplicationResult()
        if self.semantic is not None:
            similar_docs = await asyncio.to_thread(self.semantic.add, doc_id, text, content_type)

            if similar_docs:
                self.stats["semantic_duplicates"] += 1
                max_similarity = max(doc[1] for doc in similar_docs) if similar_docs else 0
                # Convert doc IDs to valid UUIDs
                duplicate_ids = []
                for doc_id, _ in similar_docs:
                    try:
                        duplicate_ids.append(UUID(doc_id))
                    except ValueError:
                        # If not a valid UUID, generate one from the doc ID using secure hash
                        import hashlib

                        doc_hash = hashlib.sha256(doc_id.encode()).hexdigest()
                        duplicate_ids.append(UUID(hex=doc_hash[:32]))

                result.is_duplicate = True
                result.duplicate_type = DuplicateType.SEMANTIC_SIMILAR.value
                result.semantic_similarity = max_similarity
                result.semantic_clusters = duplicate_ids
                result.confidence_score = max_similarity
        return result

    async def _check_fuzzy(self, doc_id: str, text: str, content_type: str) -> DuplicationResult:
        """Check for fuzzy matches."""
        result = DuplicationResult()
        if self.fuzzy is not None:
            similar_docs = await asyncio.to_thread(self.fuzzy.add, doc_id, text, content_type)

            if similar_docs:
                self.stats["fuzzy_matches"] += 1
                max_similarity = max(doc[1] for doc in similar_docs) if similar_docs else 0

                match_types = [doc[2] for doc in similar_docs]

                if "near_exact" in match_types:
                    dup_type = DuplicateType.NEAR_DUPLICATE.value
                elif "contains" in match_types or "contained_in" in match_types:
                    dup_type = DuplicateType.PARTIAL_OVERLAP.value
                else:
                    dup_type = DuplicateType.FUZZY_MATCH.value

                result.is_duplicate = True
                result.duplicate_type = dup_type
                result.confidence_score = max_similarity
        return result

    def _combine_results(self, results: List[DuplicationResult]) -> DuplicationResult:
        """Combine results from multiple levels."""
        # Filter out non-duplicate results
        valid_results = [r for r in results if r.is_duplicate]

        if not valid_results:
            return DuplicationResult()  # Return a default non-duplicate result

        # Sort by confidence score (higher is better)
        valid_results.sort(key=lambda x: x.confidence_score, reverse=True)
        return valid_results[0]

    def _determine_content_type(self, metadata: ContentMetadata) -> str:
        """Determine content type from extracted content."""
        return metadata.domain_type.value

    async def check_batch(self, contents: List[ExtractedContent]) -> List[DuplicationResult]:
        """
        Check multiple contents for duplicates efficiently.

        Args:
            contents: List of extracted contents

        Returns:
            List of DuplicationResult objects
        """
        # This method needs to be updated to also accept metadata
        raise NotImplementedError("check_batch needs to be updated to accept metadata alongside content.")

    def get_statistics(self) -> Dict[str, Any]:
        """Get deduplication statistics."""
        stats = self.stats.copy()

        # Add component-specific stats
        bloom_stats = self.bloom_filter.get_stats()
        stats["bloom_stats"] = bloom_stats

        if self.minhash is not None:
            minhash_stats = self.minhash.get_stats()
            stats["minhash_stats"] = minhash_stats

        if self.semantic is not None:
            semantic_stats = self.semantic.get_stats()
            stats["semantic_stats"] = semantic_stats

        if self.fuzzy is not None:
            fuzzy_stats = self.fuzzy.get_stats()
            stats["fuzzy_stats"] = fuzzy_stats

        # Calculate rates
        if stats["total_processed"] > 0:
            stats["duplicate_rate"] = 1 - (stats["unique_documents"] / stats["total_processed"])
            stats["avg_processing_time_ms"] = (stats["processing_time"] / stats["total_processed"]) * 1000

        return stats

    def save_state(self) -> None:
        """Save all deduplication indices to disk."""
        logger.info("Saving deduplication state...")

        # Save each component
        self.bloom_filter.save_all()

        if self.minhash is not None:
            self.minhash.save_index()

        if self.semantic is not None:
            self.semantic.save_index()

        if self.fuzzy is not None:
            self.fuzzy.save_index()

        # Save statistics
        import pickle

        stats_path = self.config.storage_path / "stats.pkl"
        with open(stats_path, "wb") as f:
            pickle.dump(self.stats, f)

        logger.info("Deduplication state saved")

    def load_state(self) -> None:
        """Load deduplication indices from disk."""
        logger.info("Loading deduplication state...")

        # Components automatically load on init
        # Load statistics
        import pickle

        stats_path = self.config.storage_path / "stats.pkl"
        if stats_path.exists():
            with open(stats_path, "rb") as f:
                self.stats = pickle.load(f)

        logger.info("Deduplication state loaded")

    def clear_duplicates(self, threshold: float = 0.9) -> None:
        """Clear stored duplicates below threshold."""
        logger.info(f"Clearing duplicates below threshold {threshold}")
        # Note: Implementation would require component-specific methods
        logger.warning("Clear duplicates requires component-specific implementation")

    async def build_bloom_filter(
        self,
        content_hashes: List[str],
        *,
        false_positive_rate: float = 0.01,
    ) -> None:
        """Build bloom filter from existing content hashes."""
        for content_hash in content_hashes:
            self.bloom_filter.add(content_hash)

    async def compute_minhash(self, text: str) -> Any:
        """Compute MinHash for text content."""
        if not HAS_NUMPY:
            raise ImportError("NumPy is required for MinHash computation")

        if self.minhash is not None:
            # Use the component's actual method signature
            return self.minhash.compute_minhash(text)
        return None

    async def compute_embedding(
        self,
        text: str,
        *,
        use_gpu: bool = True,
        batch_size: int = 32,
    ) -> Any:
        """Compute semantic embedding for text content."""
        if not HAS_NUMPY:
            raise ImportError("NumPy is required for embedding computation")

        if self.semantic is not None:
            # Use the component's actual method signature
            return self.semantic.compute_embedding([text])
        return None

    async def find_similar_content(
        self,
        embedding: Any,
        threshold: float = 0.9,
    ) -> List[Tuple[UUID, float]]:
        """Find content similar to given embedding."""
        if self.semantic is not None:
            # Use the component's actual method signature with content_type fallback
            similar = self.semantic.find_similar_content(embedding, "general", k=10)
            # Convert results to UUID format
            results = []
            for doc_id, score in similar:
                try:
                    results.append((UUID(doc_id), score))
                except ValueError:
                    # Generate UUID from doc_id hash
                    import hashlib

                    doc_hash = hashlib.sha256(str(doc_id).encode()).hexdigest()
                    results.append((UUID(hex=doc_hash[:32]), score))
            return results
        return []

    def __del__(self) -> None:
        """Cleanup resources."""
        try:
            if hasattr(self, "bloom_filter"):
                self.save_state()
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")

    async def is_duplicate(self, text: str) -> bool:
        """Simple duplicate check interface."""
        # Create minimal metadata for compatibility
        from ..protocols import ContentMetadata, DomainType

        metadata = ContentMetadata(url="unknown", domain_type=DomainType.GENERAL, title="")

        # Create minimal content
        content = ExtractedContent(text=text)

        result = await self.check_duplicates(content, metadata)
        return result.is_duplicate
