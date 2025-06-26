"""
GPU-accelerated semantic deduplication using sentence embeddings.

Implements Level 3 deduplication with sentence-transformers and FAISS,
featuring adaptive GPU memory management and CPU fallback for Pi deployment.
"""

import gc
import logging
import pickle
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any, Union, TYPE_CHECKING
from dataclasses import dataclass

# ML library imports with graceful fallbacks (proven pattern)
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
    import torch  # type: ignore[import-not-found]
    HAS_TORCH = True
except ImportError:
    if TYPE_CHECKING:
        import torch  # type: ignore[import-not-found]
    else:
        torch = None
    HAS_TORCH = False

try:
    from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    if TYPE_CHECKING:
        from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
    else:
        SentenceTransformer = None
    HAS_SENTENCE_TRANSFORMERS = False
    
try:
    import faiss  # type: ignore[import-not-found]
    HAS_FAISS = True
except ImportError:
    if TYPE_CHECKING:
        import faiss  # type: ignore[import-not-found]
    else:
        faiss = None
    HAS_FAISS = False

from functools import lru_cache
from ..protocols import HardwareCapabilities

logger = logging.getLogger(__name__)


@dataclass
class SemanticConfig:
    """Configuration for semantic deduplication."""
    model_name: str = "all-MiniLM-L6-v2"  # Efficient model, 384 dimensions
    batch_size: int = 64
    use_gpu: bool = True
    gpu_memory_fraction: float = 0.8
    similarity_threshold: float = 0.85
    index_type: str = "IVF"  # IVF, Flat, or HNSW
    nlist: int = 100  # Number of clusters for IVF
    nprobe: int = 10  # Number of clusters to search
    storage_path: Path = Path("data/semantic_index")
    use_bfloat16: bool = True
    normalize_embeddings: bool = True
    max_gpu_batch_size: int = 128  # Maximum batch size for GPU
    content_type_thresholds: Optional[Dict[str, float]] = None
    similarity_thresholds: Optional[Dict[str, float]] = None
    
    def __post_init__(self) -> None:
        if self.content_type_thresholds is None:
            self.content_type_thresholds = {
                "medical": 0.9,    # Higher threshold for medical content
                "legal": 0.88,     # Legal documents need high precision
                "technical": 0.85, # Technical blogs/docs
                "ecommerce": 0.8,  # Product descriptions
                "general": 0.85    # Default threshold
            }
        if self.similarity_thresholds is None:
            self.similarity_thresholds = {
                "very_high": 0.95,  # Nearly identical
                "high": 0.85,       # Highly similar
                "medium": 0.75,     # Moderately similar  
                "low": 0.65         # Somewhat similar
            }


class SemanticDeduplicator:
    """
    GPU-accelerated semantic deduplication with FAISS.
    
    Features:
    - Adaptive GPU memory management
    - CPU fallback for Raspberry Pi
    - IVF indexing for scalability
    - Configurable thresholds per content type
    - Incremental index updates
    - Mixed precision support
    """
    
    def __init__(
        self,
        config: SemanticConfig,
        hardware: Optional[HardwareCapabilities] = None
    ) -> None:
        self.config = config
        self.hardware = hardware or HardwareCapabilities()
        
        # Check dependencies
        if not HAS_SENTENCE_TRANSFORMERS:
            raise ImportError("sentence-transformers not installed")
        if not HAS_FAISS:
            raise ImportError("faiss not installed")
        if not HAS_TORCH:
            raise ImportError("torch not installed")
        if not HAS_NUMPY:
            raise ImportError("numpy not installed")
        
        # Detect GPU availability and adjust settings
        self._configure_hardware()
        
        # Initialize model
        self._init_model()
        
        # Initialize FAISS index
        self.index: Optional[Any] = None  # FAISS Index object
        self.id_map: Dict[int, str] = {}  # FAISS idx -> doc_id
        self.doc_id_to_idx: Dict[str, int] = {}  # doc_id -> FAISS idx
        self.embeddings_cache: Dict[str, Any] = {}  # Embeddings cache
        
        # Create storage directory
        self.config.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Load existing index if available
        self._load_index()
        
        # Statistics
        self.total_documents = 0
        self.gpu_memory_used = 0
        
        logger.info(
            f"Initialized SemanticDeduplicator with model '{self.config.model_name}', "
            f"GPU: {self.use_gpu}, batch_size: {self.config.batch_size}"
        )
    
    def _configure_hardware(self) -> None:
        """Configure hardware settings based on available resources."""
        self.use_gpu = False
        self.device = torch.device("cpu") if HAS_TORCH else "cpu"
        
        if self.config.use_gpu and HAS_TORCH and torch.cuda.is_available():
            # Check GPU memory
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            available_memory = gpu_memory * self.config.gpu_memory_fraction
            
            # Estimate memory requirements (rough estimate)
            model_memory = 500 * 1024 * 1024  # ~500MB for model
            batch_memory = self.config.batch_size * 384 * 4 * 100  # embeddings
            
            if available_memory > model_memory + batch_memory:
                self.use_gpu = True
                self.device = torch.device("cuda")
                
                # Adjust batch size based on GPU memory
                if gpu_memory < 4 * 1024**3:  # Less than 4GB
                    self.config.batch_size = min(32, self.config.batch_size)
                elif gpu_memory < 8 * 1024**3:  # Less than 8GB
                    self.config.batch_size = min(64, self.config.batch_size)
                
                logger.info(f"GPU enabled with {gpu_memory / 1024**3:.1f}GB memory")
            else:
                logger.warning("Insufficient GPU memory, falling back to CPU")
        
        # Raspberry Pi optimizations
        if self.hardware.cpu_cores <= 4 and self.hardware.total_memory_gb <= 8:
            self.config.batch_size = min(16, self.config.batch_size)
            self.config.model_name = "all-MiniLM-L6-v2"  # Force efficient model
            logger.info("Raspberry Pi mode: reduced batch size and model")
    
    def _init_model(self) -> None:
        """Initialize sentence transformer model."""
        if not HAS_SENTENCE_TRANSFORMERS:
            raise ImportError("sentence-transformers not available")
            
        logger.info(f"Loading model '{self.config.model_name}'...")
        
        self.model = SentenceTransformer(
            self.config.model_name,
            device=str(self.device)
        )
        
        # Enable mixed precision if requested and available
        if self.use_gpu and self.config.use_bfloat16 and HAS_TORCH:
            if torch.cuda.is_bf16_supported():
                self.model = self.model.half()
                logger.info("Enabled bfloat16 precision")
        
        # Get embedding dimension
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        # Warm up model
        self.model.encode(["warmup"], batch_size=1)
    
    def _create_index(self) -> None:
        """Create FAISS index based on configuration."""
        if not HAS_FAISS:
            raise ImportError("FAISS not available")
            
        if self.config.index_type == "Flat":
            # Exact search
            if self.config.normalize_embeddings:
                self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product
            else:
                self.index = faiss.IndexFlatL2(self.embedding_dim)  # L2 distance
        
        elif self.config.index_type == "IVF":
            # Inverted file index for scalability
            quantizer = faiss.IndexFlatL2(self.embedding_dim)
            self.index = faiss.IndexIVFFlat(
                quantizer,
                self.embedding_dim,
                self.config.nlist,
                faiss.METRIC_L2 if not self.config.normalize_embeddings else faiss.METRIC_INNER_PRODUCT
            )
            if hasattr(self.index, 'nprobe'):
                self.index.nprobe = self.config.nprobe
        
        elif self.config.index_type == "HNSW":
            # Hierarchical navigable small world graphs
            self.index = faiss.IndexHNSWFlat(
                self.embedding_dim,
                32,  # Number of neighbors
                faiss.METRIC_L2 if not self.config.normalize_embeddings else faiss.METRIC_INNER_PRODUCT
            )
        
        else:
            raise ValueError(f"Unknown index type: {self.config.index_type}")
        
        # Move to GPU if available and supported
        if self.use_gpu and HAS_FAISS and hasattr(faiss, 'StandardGpuResources'):
            try:
                res = faiss.StandardGpuResources()
                res.setTempMemory(int(self.config.gpu_memory_fraction * 1024 * 1024 * 1024))
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                logger.info("FAISS index moved to GPU")
            except Exception as e:
                logger.warning(f"Failed to move FAISS to GPU: {e}")
    
    def compute_embedding(
        self,
        texts: List[str],
        show_progress: bool = False
    ) -> Any:
        """Encode texts to embeddings with adaptive batching."""
        if not HAS_NUMPY:
            raise ImportError("NumPy not available")
            
        embeddings = []
        
        # Adaptive batch size based on text length
        avg_length = np.mean([len(text) for text in texts])
        if avg_length > 1000:  # Long texts
            batch_size = max(1, self.config.batch_size // 4)
        else:
            batch_size = self.config.batch_size
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                # Encode batch
                batch_embeddings = self.model.encode(
                    batch,
                    batch_size=len(batch),
                    show_progress_bar=show_progress,
                    convert_to_tensor=True,
                    normalize_embeddings=self.config.normalize_embeddings
                )
                
                # Convert to numpy
                if HAS_TORCH and torch.is_tensor(batch_embeddings):
                    batch_embeddings = batch_embeddings.cpu().numpy()
                
                embeddings.append(batch_embeddings)
                
                # Memory management
                if self.use_gpu and HAS_TORCH and i % (batch_size * 10) == 0:
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                if HAS_TORCH and "out of memory" in str(e).lower():
                    logger.warning("GPU OOM, reducing batch size")
                    # Recursively try with smaller batch
                    if batch_size > 1:
                        smaller_batch = self.compute_embedding(
                            batch,
                            show_progress=False
                        )
                        embeddings.append(smaller_batch)
                    else:
                        raise
                else:
                    raise
        
        return np.vstack(embeddings) if embeddings else np.array([])
    
    def add(
        self,
        doc_id: str,
        text: str,
        content_type: str = "general"
    ) -> List[Tuple[str, float]]:
        """
        Add document and find similar ones.
        
        Returns:
            List of (similar_doc_id, similarity_score) tuples
        """
        # Encode text
        embedding = self.compute_embedding([text])[0]
        
        # Query for similar documents before adding
        similar_docs = self.find_similar_content(
            embedding,
            content_type,
            k=10
        )
        
        # Add to index
        self._add_to_index(doc_id, embedding)
        
        return similar_docs
    
    def _add_to_index(self, doc_id: str, embedding: Any) -> None:
        """Add embedding to FAISS index."""
        if self.index is None:
            self._create_index()
            
        # Train index if needed (IVF) and not yet trained
        if self.index is not None and hasattr(self.index, 'is_trained') and not self.index.is_trained:
            self.embeddings_cache[doc_id] = embedding
            
            # Check if we have enough data to train
            if len(self.embeddings_cache) >= self.config.nlist * 40:
                logger.info(f"Training FAISS index with {len(self.embeddings_cache)} vectors...")
                training_data = np.array(list(self.embeddings_cache.values()), dtype='float32')
                self.index.train(training_data)
                
                # Add all cached embeddings to the now-trained index
                ids, vectors = zip(*self.embeddings_cache.items())
                self.index.add(np.array(vectors, dtype='float32'))
                
                # Update mappings
                for i, current_id in enumerate(ids):
                    faiss_idx = self.index.ntotal - len(ids) + i
                    self.id_map[faiss_idx] = current_id
                    self.doc_id_to_idx[current_id] = faiss_idx

                self.embeddings_cache.clear()
                logger.info("FAISS index training complete.")
            return

        # Add to the trained index
        if self.index is not None:
            idx = self.index.ntotal
            self.index.add(np.array([embedding], dtype='float32'))
            self.id_map[idx] = doc_id
            self.doc_id_to_idx[doc_id] = idx
            self.total_documents += 1
    
    def find_similar_content(
        self,
        embedding: Any,
        content_type: str,
        k: int = 10
    ) -> List[Tuple[str, float]]:
        """Search for similar embeddings."""
        if self.index is None or len(self.id_map) == 0:
            return []
        
        # Get threshold for content type
        threshold = self.config.content_type_thresholds.get(
            content_type,
            self.config.similarity_threshold
        ) if self.config.content_type_thresholds else self.config.similarity_threshold
        
        # Search
        k = min(k, len(self.id_map))
        distances, indices = self.index.search(np.array([embedding], dtype=np.float32), k)
        
        # Convert to similarities and filter
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:  # Invalid index
                continue
                
            # Convert distance to similarity
            if self.config.normalize_embeddings:
                similarity = float(dist)  # Inner product = cosine similarity
            else:
                similarity = 1.0 / (1.0 + float(dist))  # Convert L2 to similarity
            
            if similarity >= threshold:
                doc_id = self.id_map.get(idx)
                if doc_id:
                    results.append((doc_id, similarity))
        
        return results
    
    def add_batch(
        self,
        documents: List[Tuple[str, str, str]]
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Add multiple documents efficiently.
        
        Args:
            documents: List of (doc_id, text, content_type) tuples
            
        Returns:
            Dictionary mapping doc_id to similar documents
        """
        if not documents:
            return {}
        
        # Extract components
        doc_ids = [d[0] for d in documents]
        texts = [d[1] for d in documents]
        content_types = [d[2] if len(d) > 2 else "general" for d in documents]
        
        # Encode all texts
        logger.info(f"Encoding {len(texts)} documents...")
        embeddings = self.compute_embedding(texts, show_progress=True)
        
        # Find similar documents for each
        results = {}
        for doc_id, embedding, content_type in zip(doc_ids, embeddings, content_types):
            similar_docs = self.find_similar_content(embedding, content_type)
            results[doc_id] = similar_docs
            self._add_to_index(doc_id, embedding)
        
        return results
    
    def remove(self, doc_id: str) -> bool:
        """Remove document from index (requires index rebuild)."""
        if doc_id in self.doc_id_to_idx:
            # Mark for removal (actual removal requires rebuild)
            del self.doc_id_to_idx[doc_id]
            # Note: Full removal requires index rebuild
            logger.warning(f"Document {doc_id} marked for removal. Rebuild required.")
            return True
        return False
    
    def save_index(self) -> None:
        """Save FAISS index and metadata to disk."""
        if self.index is None or not HAS_FAISS:
            return
        
        # Save FAISS index
        index_path = self.config.storage_path / "faiss.index"
        if hasattr(self.index, 'index') and hasattr(faiss, 'index_gpu_to_cpu'):
            # Move GPU index to CPU for saving
            cpu_index = faiss.index_gpu_to_cpu(self.index)
            faiss.write_index(cpu_index, str(index_path))
        else:
            faiss.write_index(self.index, str(index_path))
        
        # Save metadata
        metadata = {
            'id_map': self.id_map,
            'doc_id_to_idx': self.doc_id_to_idx,
            'total_documents': self.total_documents,
            'config': self.config,
            'embedding_dim': self.embedding_dim
        }
        
        metadata_path = self.config.storage_path / "metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info(f"Saved semantic index with {self.total_documents} documents")
    
    def _load_index(self) -> None:
        """Load existing FAISS index from disk."""
        if not HAS_FAISS:
            return
            
        index_path = self.config.storage_path / "faiss.index"
        if index_path.exists():
            try:
                self.index = faiss.read_index(str(index_path))
                logger.info(f"Loaded semantic index with {self.index.ntotal} vectors")
            except Exception as e:
                logger.warning(f"Failed to load semantic index: {e}")
                self.index = None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get deduplication statistics."""
        stats: Dict[str, Any] = {
            'total_documents': self.total_documents,
            'embedding_dim': self.embedding_dim,
            'index_type': self.config.index_type,
            'device': str(self.device),
            'model': self.config.model_name,
            'batch_size': self.config.batch_size,
        }
        
        if self.use_gpu and HAS_TORCH:
            stats['gpu_memory_allocated_mb'] = torch.cuda.memory_allocated() / 1024**2
            stats['gpu_memory_reserved_mb'] = torch.cuda.memory_reserved() / 1024**2
        
        if self.index is not None:
            if hasattr(self.index, 'ntotal'):
                stats['index_size'] = self.index.ntotal
            if hasattr(self.index, 'is_trained'):
                stats['index_trained'] = self.index.is_trained
        
        return stats
    
    def __len__(self) -> int:
        """Return number of documents in index."""
        return self.total_documents 