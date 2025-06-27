"""
Fuzzy matching for partial content overlap detection.

Implements Level 4 deduplication using edit distance and other string
similarity algorithms for catching variations and partial matches.
"""

import difflib
import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np  # type: ignore[import-not-found]
from rapidfuzz import distance, fuzz  # type: ignore[import-not-found]

from ..protocols import HardwareCapabilities

logger = logging.getLogger(__name__)


@dataclass
class FuzzyConfig:
    """Configuration for fuzzy matching."""

    min_similarity: float = 0.85
    algorithm: str = "token_sort_ratio"  # Options: ratio, partial_ratio, token_sort_ratio, token_set_ratio
    max_edit_distance: int = 10
    chunk_size: int = 1000  # Characters per chunk for long documents
    overlap_size: int = 100  # Overlap between chunks
    use_phonetic: bool = False  # Use phonetic matching
    storage_path: Path = Path("data/fuzzy_index")
    content_type_thresholds: Optional[Dict[str, float]] = None

    def __post_init__(self) -> None:
        if self.content_type_thresholds is None:
            self.content_type_thresholds = {
                "medical": 0.92,  # Very high precision for medical
                "legal": 0.90,  # High precision for legal
                "technical": 0.87,  # Technical documentation
                "ecommerce": 0.83,  # Product descriptions can vary more
                "general": 0.85,  # Default threshold
            }


class FuzzyMatcher:
    """
    Advanced fuzzy matching for partial content overlap.

    Features:
    - Multiple similarity algorithms
    - Chunked processing for long documents
    - Edit distance calculations
    - Token-based matching
    - Configurable thresholds per content type
    """

    def __init__(self, config: FuzzyConfig, hardware: Optional[HardwareCapabilities] = None):
        self.config = config
        self.hardware = hardware or HardwareCapabilities()

        # Storage for document chunks and metadata
        self.documents: Dict[str, str] = {}
        self.document_chunks: Dict[str, List[str]] = {}
        self.chunk_index: Dict[str, Set[str]] = defaultdict(set)  # chunk -> doc_ids

        # Create storage directory
        self.config.storage_path.mkdir(parents=True, exist_ok=True)

        # Statistics
        self.total_comparisons = 0
        self.fuzzy_matches = 0

        logger.info(
            f"Initialized FuzzyMatcher with algorithm '{self.config.algorithm}', "
            f"min_similarity: {self.config.min_similarity}"
        )

    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for fuzzy matching."""
        # Normalize whitespace
        text = " ".join(text.split())

        # Remove extra punctuation but keep sentence structure
        text = re.sub(r"[^\w\s\.\,\!\?]", " ", text)

        # Normalize case for comparison
        text = text.lower()

        return text

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        if len(text) <= self.config.chunk_size:
            return [text]

        chunks = []
        for i in range(0, len(text), self.config.chunk_size - self.config.overlap_size):
            chunk = text[i : i + self.config.chunk_size]
            if chunk:
                chunks.append(chunk)

        return chunks

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts using configured algorithm."""
        if self.config.algorithm == "ratio":
            return float(fuzz.ratio(text1, text2) / 100.0)
        elif self.config.algorithm == "partial_ratio":
            return float(fuzz.partial_ratio(text1, text2) / 100.0)
        elif self.config.algorithm == "token_sort_ratio":
            return float(fuzz.token_sort_ratio(text1, text2) / 100.0)
        elif self.config.algorithm == "token_set_ratio":
            return float(fuzz.token_set_ratio(text1, text2) / 100.0)
        elif self.config.algorithm == "edit_distance":
            # Normalized edit distance
            max_len = max(len(text1), len(text2))
            if max_len == 0:
                return 1.0
            dist = distance.Levenshtein.distance(text1, text2)
            return float(1.0 - (dist / max_len))
        else:
            raise ValueError(f"Unknown algorithm: {self.config.algorithm}")

    def add(self, doc_id: str, text: str, content_type: str = "general") -> List[Tuple[str, float, str]]:
        """
        Add document and find fuzzy matches.

        Returns:
            List of (similar_doc_id, similarity_score, match_type) tuples
        """
        # Preprocess text
        processed_text = self._preprocess_text(text)

        # Find similar documents before adding
        similar_docs = self._find_similar(processed_text, content_type)

        # Store document
        self.documents[doc_id] = processed_text

        # Chunk and index
        chunks = self._chunk_text(processed_text)
        self.document_chunks[doc_id] = chunks

        # Add chunks to index
        for chunk in chunks:
            chunk_key = chunk[:50]  # Use first 50 chars as key
            self.chunk_index[chunk_key].add(doc_id)

        return similar_docs

    def _find_similar(self, text: str, content_type: str) -> List[Tuple[str, float, str]]:
        """Find documents with fuzzy similarity to given text."""
        if not self.documents:
            return []

        # Get threshold for content type
        if self.config.content_type_thresholds is not None:
            threshold = self.config.content_type_thresholds.get(content_type, self.config.min_similarity)
        else:
            threshold = self.config.min_similarity

        results = []
        chunks = self._chunk_text(text)

        # Find candidate documents using chunk index
        candidates = set()
        for chunk in chunks:
            chunk_key = chunk[:50]
            # Look for similar chunk keys
            for existing_key in self.chunk_index:
                if self._calculate_similarity(chunk_key, existing_key) > 0.8:
                    candidates.update(self.chunk_index[existing_key])

        # If no candidates from chunks, check all documents (for small datasets)
        if not candidates and len(self.documents) < 1000:
            candidates = set(self.documents.keys())

        # Calculate similarities with candidates
        for doc_id in candidates:
            if doc_id not in self.documents:
                continue

            # Full document comparison
            similarity = self._calculate_similarity(text, self.documents[doc_id])
            self.total_comparisons += 1

            if similarity >= threshold:
                match_type = self._classify_match(similarity, text, self.documents[doc_id])
                results.append((doc_id, similarity, match_type))
                self.fuzzy_matches += 1

        # Sort by similarity
        results.sort(key=lambda x: x[1], reverse=True)

        return results

    def _classify_match(self, similarity: float, text1: str, text2: str) -> str:
        """Classify the type of fuzzy match."""
        if similarity >= 0.95:
            return "near_exact"
        elif similarity >= 0.90:
            return "very_similar"
        elif len(text1) > len(text2) * 1.5:
            return "contains"
        elif len(text2) > len(text1) * 1.5:
            return "contained_in"
        else:
            return "similar"

    def find_partial_matches(self, query: str, min_overlap: int = 100) -> List[Tuple[str, float, int, int]]:
        """
        Find documents containing partial matches of the query.

        Returns:
            List of (doc_id, similarity, start_pos, end_pos) tuples
        """
        query = self._preprocess_text(query)
        results = []

        for doc_id, doc_text in self.documents.items():
            # Use difflib to find matching blocks
            matcher = difflib.SequenceMatcher(None, query, doc_text)

            for match in matcher.get_matching_blocks():
                if match.size >= min_overlap:
                    # Calculate local similarity
                    match_text = doc_text[match.b : match.b + match.size]
                    similarity = self._calculate_similarity(query[match.a : match.a + match.size], match_text)

                    if similarity >= self.config.min_similarity:
                        results.append((doc_id, similarity, match.b, match.b + match.size))

        return results

    def find_near_duplicates_batch(
        self, documents: List[Tuple[str, str, str]]
    ) -> Dict[str, List[Tuple[str, float, str]]]:
        """
        Find near duplicates within a batch of documents.

        Args:
            documents: List of (doc_id, text, content_type) tuples

        Returns:
            Dictionary mapping doc_id to list of similar documents
        """
        results = {}

        # Preprocess all documents
        processed_docs = []
        for doc_id, text, content_type in documents:
            processed = self._preprocess_text(text)
            processed_docs.append((doc_id, processed, content_type))

        # Compare all pairs (optimization: use blocking or LSH for large batches)
        for i, (doc_id1, text1, type1) in enumerate(processed_docs):
            similar_docs = []

            # Check against existing documents
            existing_similar = self._find_similar(text1, type1)
            similar_docs.extend(existing_similar)

            # Check against other documents in batch
            for _j, (doc_id2, text2, type2) in enumerate(processed_docs[i + 1 :], i + 1):
                similarity = self._calculate_similarity(text1, text2)
                if self.config.content_type_thresholds is not None:
                    threshold = min(
                        self.config.content_type_thresholds.get(type1, self.config.min_similarity),
                        self.config.content_type_thresholds.get(type2, self.config.min_similarity),
                    )
                else:
                    threshold = self.config.min_similarity

                if similarity >= threshold:
                    match_type = self._classify_match(similarity, text1, text2)
                    similar_docs.append((doc_id2, similarity, match_type))

            results[doc_id1] = similar_docs

            # Add to index
            self.documents[doc_id1] = text1
            chunks = self._chunk_text(text1)
            self.document_chunks[doc_id1] = chunks
            for chunk in chunks:
                chunk_key = chunk[:50]
                self.chunk_index[chunk_key].add(doc_id1)

        return results

    def remove(self, doc_id: str) -> bool:
        """Remove document from fuzzy matcher."""
        if doc_id in self.documents:
            # Remove from main storage
            del self.documents[doc_id]

            # Remove from chunk index
            if doc_id in self.document_chunks:
                for chunk in self.document_chunks[doc_id]:
                    chunk_key = chunk[:50]
                    if chunk_key in self.chunk_index:
                        self.chunk_index[chunk_key].discard(doc_id)
                        if not self.chunk_index[chunk_key]:
                            del self.chunk_index[chunk_key]

                del self.document_chunks[doc_id]

            return True
        return False

    def get_stats(self) -> Dict[str, Any]:
        """Get fuzzy matching statistics."""
        avg_doc_length = 0
        if self.documents:
            avg_doc_length = np.mean([len(doc) for doc in self.documents.values()])

        match_rate = 0.0
        if self.total_comparisons > 0:
            match_rate = self.fuzzy_matches / self.total_comparisons

        return {
            "total_documents": len(self.documents),
            "total_chunks": sum(len(chunks) for chunks in self.document_chunks.values()),
            "unique_chunk_keys": len(self.chunk_index),
            "avg_document_length": avg_doc_length,
            "total_comparisons": self.total_comparisons,
            "fuzzy_matches": self.fuzzy_matches,
            "match_rate": match_rate,
            "algorithm": self.config.algorithm,
            "min_similarity": self.config.min_similarity,
        }

    def save_index(self) -> None:
        """Save fuzzy matcher state to disk."""
        import pickle

        state = {
            "documents": self.documents,
            "document_chunks": self.document_chunks,
            "chunk_index": dict(self.chunk_index),
            "config": self.config,
            "stats": {
                "total_comparisons": self.total_comparisons,
                "fuzzy_matches": self.fuzzy_matches,
            },
        }

        index_path = self.config.storage_path / "fuzzy_index.pkl"
        with open(index_path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info(f"Saved fuzzy index with {len(self.documents)} documents")

    def load_index(self) -> None:
        """Load fuzzy matcher state from disk."""
        import pickle

        index_path = self.config.storage_path / "fuzzy_index.pkl"
        if index_path.exists():
            try:
                with open(index_path, "rb") as f:
                    state = pickle.load(f)

                self.documents = state["documents"]
                self.document_chunks = state["document_chunks"]
                self.chunk_index = defaultdict(set, state["chunk_index"])

                if "stats" in state:
                    self.total_comparisons = state["stats"]["total_comparisons"]
                    self.fuzzy_matches = state["stats"]["fuzzy_matches"]

                logger.info(f"Loaded fuzzy index with {len(self.documents)} documents")

            except Exception as e:
                logger.warning(f"Failed to load fuzzy index: {e}")

    def __len__(self) -> int:
        """Return number of documents."""
        return len(self.documents)
