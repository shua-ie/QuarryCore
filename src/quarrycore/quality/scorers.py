"""
GPU-aware quality scorers with automatic CPU fallback.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import pkg_resources
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

try:
    import fasttext

    HAS_FASTTEXT = True
except ImportError:
    HAS_FASTTEXT = False
    fasttext = None

logger = structlog.get_logger(__name__)


@dataclass
class Score:
    """Score result from a scorer."""

    name: str
    value: float


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


class LengthScorer:
    """Scores content based on length."""

    name = "length"
    WEIGHT = 0.3

    async def __call__(self, text: str) -> Score:
        """Score based on text length."""
        # Score 1.0 if text is longer than 400 characters
        score_value = 1.0 if len(text) > 400 else 0.0
        return Score(self.name, score_value)


class LanguageScorer:
    """Scores content based on language detection."""

    name = "language"
    WEIGHT = 0.4
    _lid = None

    def __init__(self):
        """Initialize language scorer with lazy loading."""
        self._lid_lock = asyncio.Lock()

    async def _ensure_model_loaded(self):
        """Lazy load the language detection model."""
        async with self._lid_lock:
            if self._lid is None:
                if _TEST_MODE:
                    logger.info("Running in test mode, using mock language model")
                    self._lid = MockLanguageModel()
                elif not HAS_FASTTEXT:
                    logger.warning("FastText not available, using mock model")
                    self._lid = MockLanguageModel()
                else:
                    try:
                        # Try to load from package resources
                        model_path = pkg_resources.resource_filename("quarrycore.resources", "lid.176.ftz")
                        self._lid = fasttext.load_model(model_path)
                    except Exception:
                        # Fallback to default path
                        try:
                            self._lid = fasttext.load_model("resources/lid.176.ftz")
                        except Exception:
                            logger.warning("Failed to load FastText model, using mock")
                            self._lid = MockLanguageModel()

    async def __call__(self, text: str) -> Score:
        """Score based on language detection."""
        await self._ensure_model_loaded()

        try:
            # Clean text for language detection
            clean_text = text.replace("\n", " ").strip()
            if not clean_text:
                return Score(self.name, 0.0)

            if isinstance(self._lid, MockLanguageModel):
                predictions = self._lid.predict(clean_text)
                lang = predictions[0][0].split("__")[-1] if predictions[0] else "unk"
            else:
                # FastText prediction
                predictions = self._lid.predict(clean_text, k=1)
                lang = predictions[0][0].split("__")[-1] if predictions[0] else "unk"

            # Score 1.0 if English, 0.0 otherwise
            score_value = 1.0 if lang == "en" else 0.0
            return Score(self.name, score_value)

        except Exception as e:
            logger.error("Error in language detection", error=str(e))
            return Score(self.name, 0.5)  # Default middle score on error


class MockLanguageModel:
    """Mock language model for testing."""

    def predict(self, text: str, k=1):
        """Mock prediction - returns English for reasonable text."""
        if len(text) > 20 and text.count(" ") > 2:
            # Simple heuristic: check for Spanish words
            spanish_words = {"es", "en", "con", "español", "texto", "gramática"}
            text_lower = text.lower()
            if any(word in text_lower for word in spanish_words):
                return (["__label__es"], [0.99])
            return (["__label__en"], [0.99])
        return (["__label__unknown"], [0.5])


class TransformerCoherenceScorer:
    """
    Scores text coherence using transformer models with GPU acceleration support.
    Automatically falls back to CPU if CUDA is not available.
    """

    name = "coherence"
    WEIGHT = 0.3

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

    async def __call__(self, text: str) -> Score:
        """Score the coherence of the given text."""
        score_value = await self.score(text)
        return Score(self.name, score_value)

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


# List of all available scorers
ALL_SCORERS = [
    LengthScorer(),
    LanguageScorer(),
    # TransformerCoherenceScorer needs config, will be instantiated by QualityAssessor
]
