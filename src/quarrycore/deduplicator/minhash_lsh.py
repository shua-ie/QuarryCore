"""
MinHash LSH for near-duplicate detection.

Implements Level 2 deduplication using MinHash with Locality Sensitive Hashing
for efficient Jaccard similarity computation at scale.
"""

import pickle
import logging
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Set, Any
from dataclasses import dataclass
import hashlib
import re

try:
    from datasketch import MinHash  # type: ignore[import-not-found]
    from datasketch.lsh import MinHashLSH  # type: ignore[import-not-found]
    import numpy as np  # type: ignore[import-not-found]
    HAS_DATASKETCH = True
    HAS_NUMPY = True
except ImportError:
    MinHash = None
    MinHashLSH = None
    np = None
    HAS_DATASKETCH = False
    HAS_NUMPY = False

from ..protocols import HardwareCapabilities

logger = logging.getLogger(__name__)


@dataclass
class MinHashConfig:
    """Configuration for MinHash LSH."""
    num_perm: int = 128          # Number of permutations (higher = more accurate)
    threshold: float = 0.8        # Jaccard similarity threshold
    num_bands: Optional[int] = None  # Auto-calculated if None
    storage_path: Path = Path("data/minhash_lsh")
    tokenizer: str = "word"       # 'word', 'char', or 'shingle'
    shingle_size: int = 3         # For shingle tokenizer
    lowercase: bool = True
    remove_punctuation: bool = True


class MinHashLSHDeduplicator:
    """
    MinHash LSH for scalable near-duplicate detection.
    
    Features:
    - Configurable Jaccard similarity threshold
    - Multiple tokenization strategies
    - Incremental index updates
    - Persistent storage with fast loading
    - Memory-efficient operation
    """
    
    def __init__(
        self,
        config: MinHashConfig,
        hardware: Optional[HardwareCapabilities] = None
    ):
        self.config = config
        self.hardware = hardware or HardwareCapabilities()
        
        # Calculate optimal bands/rows if not specified
        if self.config.num_bands is None:
            self.config.num_bands = self._calculate_optimal_bands()
        
        # Initialize LSH index
        self.lsh = MinHashLSH(
            threshold=self.config.threshold,
            num_perm=self.config.num_perm,
            weights=(0.5, 0.5)  # Balanced weights
        )
        
        # Storage for MinHash objects
        self.minhashes: Dict[str, MinHash] = {}
        
        # Create storage directory
        self.config.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Load existing index if available
        self._load_index()
        
        # Statistics
        self.total_documents = 0
        self.duplicate_pairs = 0
        
        logger.info(
            f"Initialized MinHashLSH with {self.config.num_perm} permutations, "
            f"threshold: {self.config.threshold}, bands: {self.config.num_bands}"
        )
    
    def _calculate_optimal_bands(self) -> int:
        """Calculate optimal number of bands for given threshold."""
        # Formula: b = ceil(log(1/num_perm) / log(threshold))
        # Simplified approximation for common cases
        if self.config.threshold >= 0.9:
            return self.config.num_perm // 8
        elif self.config.threshold >= 0.8:
            return self.config.num_perm // 6
        elif self.config.threshold >= 0.7:
            return self.config.num_perm // 4
        else:
            return self.config.num_perm // 2
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text based on configured strategy."""
        if self.config.lowercase:
            text = text.lower()
        
        if self.config.remove_punctuation:
            text = re.sub(r'[^\w\s]', ' ', text)
        
        if self.config.tokenizer == "word":
            tokens = text.split()
        elif self.config.tokenizer == "char":
            tokens = list(text)
        elif self.config.tokenizer == "shingle":
            # Character n-grams
            tokens = [
                text[i:i + self.config.shingle_size]
                for i in range(len(text) - self.config.shingle_size + 1)
            ]
        else:
            raise ValueError(f"Unknown tokenizer: {self.config.tokenizer}")
        
        return tokens
    
    def compute_minhash(self, text: str) -> MinHash:
        """Create MinHash signature for text."""
        tokens = self._tokenize(text)
        
        minhash = MinHash(num_perm=self.config.num_perm)
        for token in tokens:
            minhash.update(token.encode('utf-8'))
        
        return minhash
    
    def add(self, doc_id: str, text: str) -> List[str]:
        """
        Add document to LSH index.
        
        Returns:
            List of similar document IDs
        """
        # Create MinHash
        minhash = self.compute_minhash(text)
        
        # Query for similar documents before adding
        similar_docs = self.query(minhash)
        
        # Add to index
        self.lsh.insert(doc_id, minhash)
        self.minhashes[doc_id] = minhash
        self.total_documents += 1
        
        if similar_docs:
            self.duplicate_pairs += len(similar_docs)
        
        return similar_docs
    
    def query(self, minhash: MinHash) -> List[str]:
        """Query for similar documents."""
        return list(self.lsh.query(minhash))
    
    def query_text(self, text: str) -> List[Tuple[str, float]]:
        """
        Query for similar documents using text.
        
        Returns:
            List of (doc_id, similarity_score) tuples
        """
        query_minhash = self.compute_minhash(text)
        similar_ids = self.query(query_minhash)
        
        # Calculate exact Jaccard similarities
        results = []
        for doc_id in similar_ids:
            if doc_id in self.minhashes:
                similarity = query_minhash.jaccard(self.minhashes[doc_id])
                results.append((doc_id, similarity))
        
        # Sort by similarity
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    
    def remove(self, doc_id: str) -> bool:
        """Remove document from index."""
        if doc_id in self.minhashes:
            self.lsh.remove(doc_id)
            del self.minhashes[doc_id]
            self.total_documents -= 1
            return True
        return False
    
    def add_batch(self, documents: List[Tuple[str, str]]) -> Dict[str, List[str]]:
        """
        Add multiple documents efficiently.
        
        Args:
            documents: List of (doc_id, text) tuples
            
        Returns:
            Dictionary mapping doc_id to list of similar doc_ids
        """
        results = {}
        
        # Create MinHashes in parallel if possible
        minhashes = []
        for doc_id, text in documents:
            minhash = self.compute_minhash(text)
            minhashes.append((doc_id, minhash))
        
        # Add to index and find duplicates
        for doc_id, minhash in minhashes:
            similar_docs = self.query(minhash)
            results[doc_id] = similar_docs
            
            self.lsh.insert(doc_id, minhash)
            self.minhashes[doc_id] = minhash
            self.total_documents += 1
            
            if similar_docs:
                self.duplicate_pairs += len(similar_docs)
        
        return results
    
    def find_duplicate_clusters(self) -> List[Set[str]]:
        """Find all clusters of duplicate documents."""
        clusters = []
        processed = set()
        
        for doc_id in self.minhashes:
            if doc_id in processed:
                continue
            
            # Find all documents similar to this one
            similar_docs = self.query_text(self.minhashes[doc_id])
            if similar_docs:
                cluster = {doc_id}
                cluster.update(doc[0] for doc in similar_docs)
                clusters.append(cluster)
                processed.update(cluster)
        
        return clusters
    
    def save_index(self) -> None:
        """Save MinHash LSH index to disk."""
        import pickle
        
        data = {
            'lsh': self.lsh,
            'minhashes': self.minhashes,
            'config': self.config
        }
        
        index_path = self.config.storage_path / "minhash_index.pkl"
        with open(index_path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        logger.info(f"Saved MinHash LSH index with {len(self.minhashes)} documents")
    
    def _load_index(self) -> None:
        """Load existing MinHash LSH index from disk."""
        index_path = self.config.storage_path / "minhash_index.pkl"
        if index_path.exists():
            try:
                import pickle
                with open(index_path, 'rb') as f:
                    data = pickle.load(f)
                self.lsh = data['lsh']
                self.minhashes = data['minhashes']
                logger.info(f"Loaded MinHash LSH index with {len(self.minhashes)} documents")
            except Exception as e:
                logger.warning(f"Failed to load MinHash LSH index: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get MinHash LSH statistics."""
        return {
            'total_documents': len(self.minhashes),
            'hash_functions': self.config.num_perm,
            'bands': self.config.num_bands,
            'rows_per_band': self._calculate_optimal_bands(),
            'threshold': self.config.threshold,
            'storage_path': str(self.config.storage_path)
        }
    
    def _estimate_memory_usage(self) -> int:
        """Estimate memory usage in bytes."""
        # Each MinHash uses approximately num_perm * 8 bytes
        minhash_memory = self.total_documents * self.config.num_perm * 8
        
        # LSH index overhead (rough estimate)
        lsh_overhead = self.total_documents * 100  # ~100 bytes per document
        
        return minhash_memory + lsh_overhead
    
    def __len__(self) -> int:
        """Return number of documents in index."""
        return self.total_documents 