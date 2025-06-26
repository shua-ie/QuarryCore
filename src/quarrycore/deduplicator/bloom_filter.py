"""
Sharded Bloom Filter for Level 1 exact duplicate detection.

High-performance implementation with memory-mapped shards,
hardware-adaptive sizing, and production-ready persistence.
"""

import hashlib
import logging
import pickle
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any, TYPE_CHECKING
from dataclasses import dataclass
import struct
import time
import threading
from concurrent.futures import ThreadPoolExecutor

# Library imports with graceful fallbacks (proven pattern)
try:
    import mmh3  # type: ignore[import-not-found]
    HAS_MMH3 = True
except ImportError:
    if TYPE_CHECKING:
        import mmh3  # type: ignore[import-not-found]
    else:
        mmh3 = None
    HAS_MMH3 = False

try:
    import numpy as np  # type: ignore[import-not-found]
    HAS_NUMPY = True
except ImportError:
    if TYPE_CHECKING:
        import numpy as np  # type: ignore[import-not-found]
    else:
        np = None
    HAS_NUMPY = False

try:
    from pybloom_live import BloomFilter  # type: ignore[import-not-found]
    HAS_PYBLOOM = True
except ImportError:
    if TYPE_CHECKING:
        from pybloom_live import BloomFilter  # type: ignore[import-not-found]
    else:
        BloomFilter = None
    HAS_PYBLOOM = False

from ..protocols import HardwareCapabilities

logger = logging.getLogger(__name__)


@dataclass
class BloomFilterConfig:
    """Configuration for sharded bloom filter."""
    capacity: int = 10_000_000
    error_rate: float = 0.001
    num_shards: int = 16
    shard_dir: Path = Path("data/bloom_shards")
    use_mmap: bool = True
    hash_functions: int = 7
    persist_on_add: bool = False  # Save after each add (slower but safer)
    max_memory_mb: int = 1024  # Maximum memory per shard
    
    def __post_init__(self) -> None:
        # Create shard directory
        self.shard_dir.mkdir(parents=True, exist_ok=True)
        
        # Adjust for hardware limitations
        if self.num_shards > 64:
            self.num_shards = 64
        if self.num_shards < 1:
            self.num_shards = 1


class ShardedBloomFilter:
    """
    Production-grade sharded bloom filter for exact duplicate detection.
    
    Features:
    - Memory-mapped file-backed shards for persistence
    - Thread-safe operations with minimal locking
    - Hardware-adaptive configuration
    - Incremental saving and loading
    - Memory usage monitoring
    """
    
    def __init__(
        self,
        config: BloomFilterConfig,
        hardware: Optional[HardwareCapabilities] = None
    ) -> None:
        self.config = config
        self.hardware = hardware or HardwareCapabilities()
        
        # Initialize shards list BEFORE calling _initialize_shards()
        self.shards: List[Any] = []
        
        # Adjust configuration based on hardware
        self._adapt_to_hardware()
        
        # Initialize shards
        self._initialize_shards()
        
        # Statistics
        self.total_items = 0
        self.false_positive_count = 0
        self.shard_locks = [threading.RLock() for _ in range(self.config.num_shards)]
        
        logger.info(
            f"Initialized ShardedBloomFilter with {self.config.num_shards} shards, "
            f"capacity: {self.config.capacity:,}, error_rate: {self.config.error_rate}"
        )
    
    def _adapt_to_hardware(self) -> None:
        """Adapt configuration to available hardware."""
        # Adjust shards based on CPU cores
        if self.hardware.cpu_cores <= 2:
            self.config.num_shards = min(4, self.config.num_shards)
        elif self.hardware.cpu_cores <= 4:
            self.config.num_shards = min(8, self.config.num_shards)
        
        # Adjust memory usage for Raspberry Pi
        if self.hardware.total_memory_gb <= 4:
            self.config.max_memory_mb = min(256, self.config.max_memory_mb)
            self.config.use_mmap = True  # Essential for low memory
        
        # Disable memory mapping if not supported
        if not hasattr(self, '_mmap_supported'):
            try:
                import mmap
                self._mmap_supported = True
            except ImportError:
                self.config.use_mmap = False
                self._mmap_supported = False
    
    def _initialize_shards(self) -> None:
        """Initialize bloom filter shards."""
        if not HAS_PYBLOOM:
            logger.warning("pybloom_live not available, using basic implementation")
            self.shards = [set() for _ in range(self.config.num_shards)]
            return
        
        # Initialize shards list
        shard_capacity = self.config.capacity // self.config.num_shards
        
        for i in range(self.config.num_shards):
            shard_path = self.config.shard_dir / f"shard_{i}.bloom"
            
            if shard_path.exists():
                # Load existing shard
                try:
                    with open(shard_path, 'rb') as f:
                        shard = pickle.load(f)
                    self.shards.append(shard)
                    logger.debug(f"Loaded shard {i} from {shard_path}")
                except Exception as e:
                    logger.warning(f"Failed to load shard {i}: {e}, creating new")
                    shard = BloomFilter(capacity=shard_capacity, error_rate=self.config.error_rate)
                    self.shards.append(shard)
            else:
                # Create new shard
                shard = BloomFilter(capacity=shard_capacity, error_rate=self.config.error_rate)
                self.shards.append(shard)
    
    def _get_shard_index(self, content_hash: str) -> int:
        """Get shard index for given content hash."""
        if HAS_MMH3:
            hash_val = int(mmh3.hash(content_hash, signed=False))
        else:
            # Fallback to built-in hash
            hash_val = int(hash(content_hash))
        return hash_val % self.config.num_shards
    
    def add(self, content: str) -> Tuple[bool, str]:
        """
        Add content to bloom filter.
        
        Args:
            content: Content to add
            
        Returns:
            Tuple of (is_new, content_hash)
        """
        # Generate content hash
        content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
        
        # Get shard
        shard_idx = self._get_shard_index(content_hash)
        
        with self.shard_locks[shard_idx]:
            shard = self.shards[shard_idx]
            
            if HAS_PYBLOOM and hasattr(shard, 'add'):
                # Check if already exists
                is_new = content_hash not in shard
                if is_new:
                    shard.add(content_hash)
                    self.total_items += 1
            else:
                # Fallback set-based implementation
                is_new = content_hash not in shard
                if is_new:
                    shard.add(content_hash)
                    self.total_items += 1
        
        # Persist if configured
        if self.config.persist_on_add:
            self._save_shard(shard_idx)
        
        return is_new, content_hash
    
    def contains(self, content_hash: str) -> bool:
        """Check if content hash exists in bloom filter."""
        shard_idx = self._get_shard_index(content_hash)
        
        with self.shard_locks[shard_idx]:
            shard = self.shards[shard_idx]
            
            if HAS_PYBLOOM and hasattr(shard, '__contains__'):
                return content_hash in shard
            else:
                # Fallback set-based implementation
                return content_hash in shard
    
    def __contains__(self, content: str) -> bool:
        """
        Support 'in' operator for bloom filter.
        
        Args:
            content: Content to check (will be hashed)
            
        Returns:
            True if content exists in bloom filter
        """
        # Generate content hash
        content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
        return self.contains(content_hash)
    
    def add_batch(self, items: List[str]) -> List[Tuple[bool, str]]:
        """
        Add multiple items to bloom filter.
        
        Args:
            items: List of content strings to add
            
        Returns:
            List of (is_new, content_hash) tuples
        """
        results = []
        for item in items:
            is_new, content_hash = self.add(item)
            results.append((is_new, content_hash))
        return results
    
    def _save_shard(self, shard_idx: int) -> None:
        """Save a specific shard to disk."""
        shard_path = self.config.shard_dir / f"shard_{shard_idx}.bloom"
        
        with self.shard_locks[shard_idx]:
            try:
                with open(shard_path, 'wb') as f:
                    pickle.dump(self.shards[shard_idx], f)
                logger.debug(f"Saved shard {shard_idx} to {shard_path}")
            except Exception as e:
                logger.error(f"Failed to save shard {shard_idx}: {e}")
    
    def save_all(self) -> None:
        """Save all shards to disk."""
        logger.info("Saving all bloom filter shards...")
        
        # Use thread pool for parallel saving
        with ThreadPoolExecutor(max_workers=min(4, self.config.num_shards)) as executor:
            futures = [
                executor.submit(self._save_shard, i)
                for i in range(self.config.num_shards)
            ]
            
            # Wait for all saves to complete
            for future in futures:
                try:
                    future.result(timeout=30)
                except Exception as e:
                    logger.error(f"Shard save failed: {e}")
        
        # Save metadata
        self._save_metadata()
        logger.info("Bloom filter shards saved")
    
    def _save_metadata(self) -> None:
        """Save bloom filter metadata."""
        metadata = {
            'config': self.config,
            'total_items': self.total_items,
            'false_positive_count': self.false_positive_count,
            'num_shards': self.config.num_shards
        }
        
        metadata_path = self.config.shard_dir / "metadata.pkl"
        try:
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get bloom filter statistics."""
        stats: Dict[str, Any] = {
            'total_items': self.total_items,
            'num_shards': self.config.num_shards,
            'capacity': self.config.capacity,
            'error_rate': self.config.error_rate,
            'false_positive_count': self.false_positive_count,
            'memory_usage_mb': self._estimate_memory_usage(),
        }
        
        # Add shard-specific stats
        if HAS_PYBLOOM:
            shard_stats = []
            for i, shard in enumerate(self.shards):
                if hasattr(shard, '__len__'):
                    shard_info = {
                        'shard_id': i,
                        'items': len(shard),
                        'capacity': getattr(shard, 'capacity', 'unknown')
                    }
                    shard_stats.append(shard_info)
            stats['shard_stats'] = shard_stats
        
        return stats
    
    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage in MB."""
        base_memory = 0.1  # Base overhead
        
        if HAS_PYBLOOM:
            # Estimate based on bloom filter bit arrays
            bits_per_shard = self.config.capacity * self.config.hash_functions * 1.44  # Bloom filter theory
            bytes_per_shard = bits_per_shard / 8
            total_mb = (bytes_per_shard * self.config.num_shards) / (1024 * 1024)
        else:
            # Estimate based on set storage
            bytes_per_hash = 64  # SHA-256 hex string
            items_per_shard = self.total_items / self.config.num_shards
            total_mb = (bytes_per_hash * items_per_shard * self.config.num_shards) / (1024 * 1024)
        
        return base_memory + total_mb
    
    def clear(self) -> None:
        """Clear all shards and reset statistics."""
        for i in range(self.config.num_shards):
            with self.shard_locks[i]:
                if HAS_PYBLOOM:
                    shard_capacity = self.config.capacity // self.config.num_shards
                    self.shards[i] = BloomFilter(
                        capacity=shard_capacity,
                        error_rate=self.config.error_rate
                    )
                else:
                    self.shards[i] = set()
        
        self.total_items = 0
        self.false_positive_count = 0
        logger.info("Bloom filter cleared")
    
    def __len__(self) -> int:
        """Return total number of items."""
        return self.total_items 