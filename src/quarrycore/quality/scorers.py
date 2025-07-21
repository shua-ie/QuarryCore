"""
GPU-aware quality scorers with automatic CPU fallback.
"""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import structlog

if TYPE_CHECKING:
    import torch
    from sentence_transformers import SentenceTransformer

from quarrycore.config.config import QualityConfig
from quarrycore.observability.metrics import METRICS

# Check if we're in test mode
_TEST_MODE = os.environ.get("QUARRY_TEST_MODE", "0") == "1"

# Optional ML dependencies
try:
    import torch
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

    HAS_ML_LIBS = True
except ImportError:
    HAS_ML_LIBS = False
    torch = None
    SentenceTransformer = None
    cosine_similarity = None

logger = structlog.get_logger(__name__)


def _resolve_device(setting: str) -> torch.device:
    """Resolve the device setting to a torch device."""
    if not HAS_ML_LIBS or torch is None:
        return None

    if setting == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cpu":
            logger.warning("CUDA requested but not available, falling back to CPU")
        return device
    elif setting == "cpu":
        return torch.device("cpu")
    else:  # "auto"
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TransformerCoherenceScorer:
    """
    Scores text coherence using transformer models with GPU acceleration support.
    Automatically falls back to CPU if CUDA is not available.
    """

    def __init__(self, config: QualityConfig):
        """Initialize the scorer with device configuration."""
        self.config = config
        self._model: Optional[SentenceTransformer] = None
        self._device: Optional[torch.device] = None
        self._reference_embedding: Optional[np.ndarray] = None
        self._initialized = False
        self._init_lock = asyncio.Lock()

    async def _initialize(self) -> None:
        """Lazy initialization of the model and reference embeddings."""
        async with self._init_lock:
            if self._initialized:
                return

            if _TEST_MODE:
                logger.info("Running in test mode, using mock model")
                self._initialized = True
                return

            if not HAS_ML_LIBS:
                raise ImportError(
                    "ML libraries (torch, sentence_transformers) are required for TransformerCoherenceScorer"
                )

            # Resolve device
            self._device = _resolve_device(self.config.device)
            logger.info("Initializing TransformerCoherenceScorer", device=str(self._device))

            # Initialize model
            self._model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=self._device)
            self._model.max_seq_length = 256

            # Pre-compute reference embedding for coherent text
            reference_text = (
                "This is a well-structured paragraph that demonstrates coherent writing. "
                "It has a clear topic, logical flow, and proper grammar. The sentences "
                "connect naturally and convey meaningful information."
            )

            # Run encoding in executor to avoid blocking
            loop = asyncio.get_running_loop()
            self._reference_embedding = await loop.run_in_executor(
                None, lambda: self._model.encode(reference_text, convert_to_numpy=True, device=self._device)
            )

            self._initialized = True
            logger.info("TransformerCoherenceScorer initialized successfully")

    async def score(self, text: str) -> float:
        """
        Score the coherence of the given text.

        Args:
            text: The text to score

        Returns:
            Coherence score between 0 and 1
        """
        if not self._initialized:
            await self._initialize()

        if _TEST_MODE:
            # Return a reasonable mock score
            return 0.75

        if not text or len(text.split()) < 10:
            return 0.0

        try:
            # Measure encoding time
            with METRICS["quality_scorer_latency"].labels(scorer="transformer_coherence").time():
                loop = asyncio.get_running_loop()

                # Encode text (runs in thread pool to avoid blocking)
                text_embedding = await loop.run_in_executor(
                    None,
                    lambda: self._model.encode(
                        text[:1024], convert_to_numpy=True, device=self._device  # Limit text length for performance
                    ),
                )

                # Calculate cosine similarity
                similarity = cosine_similarity([text_embedding], [self._reference_embedding])[0][0]

                # Normalize to 0-1 range
                score = float(max(0.0, min(1.0, similarity)))

                return score

        except Exception as e:
            logger.error("Error scoring text coherence", error=str(e))
            METRICS["quality_scorer_errors"].labels(scorer="transformer_coherence").inc()
            return 0.5  # Default middle score on error

    def cleanup(self) -> None:
        """Clean up resources."""
        if self._model is not None:
            # Clear CUDA cache if using GPU
            if self._device and self._device.type == "cuda" and torch is not None:
                torch.cuda.empty_cache()
            self._model = None
            self._reference_embedding = None
            self._initialized = False
