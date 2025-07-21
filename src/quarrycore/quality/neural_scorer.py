from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING, Any, Dict, List

import structlog

if TYPE_CHECKING:
    import numpy as np  # type: ignore[import-not-found]
    import torch  # type: ignore[import-not-found]
    from detoxify import Detoxify  # type: ignore[import-not-found]
    from sentence_transformers import (
        SentenceTransformer,  # type: ignore[import-not-found]
    )
    from sentence_transformers.util import cos_sim  # type: ignore[import-not-found]

from quarrycore.protocols import ContentMetadata, ExtractedContent, QualityScore

# Setup logger
logger = structlog.get_logger(__name__)

# Check if we're in test mode
_TEST_MODE = os.environ.get("QUARRY_TEST_MODE", "0") == "1"

# Optional ML dependencies
try:
    import numpy as np  # type: ignore[import-not-found]
    import torch  # type: ignore[import-not-found]
    from detoxify import Detoxify  # type: ignore[import-not-found]
    from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
    from sentence_transformers.util import cos_sim  # type: ignore[import-not-found]

    HAS_ML_LIBS = True
except ImportError:
    HAS_ML_LIBS = False
    np = None
    torch = None
    Detoxify = None
    SentenceTransformer = None
    cos_sim = None


class MockTensor:
    """Mock tensor for testing environments without torch."""

    def __init__(self, shape: List[int]):
        self.shape = shape
        # Ensure consistent typing - always List[List[float]]
        if len(shape) >= 2:
            self._data = [[0.7 for _ in range(384)] for _ in range(shape[0])]
        else:
            # For 1D case, still create List[List[float]] structure
            self._data = [[0.7]]

    def __getitem__(self, key):
        if isinstance(key, slice):
            return MockTensor([len(self._data[key])])
        return self._data[key]

    def mean(self):
        class MockMean:
            def item(self):
                return 0.7  # Reasonable coherence score

        return MockMean()


class MockToxicityModel:
    """Mock toxicity model for testing."""

    def __init__(self, device: str = "cpu"):
        self.device = device

    def predict(self, texts: List[str]) -> Dict[str, List[float]]:
        """Mock toxicity prediction."""
        return {
            "toxicity": [0.1] * len(texts),  # Low toxicity scores
            "severe_toxicity": [0.05] * len(texts),
            "obscene": [0.05] * len(texts),
            "threat": [0.02] * len(texts),
            "insult": [0.08] * len(texts),
            "identity_attack": [0.03] * len(texts),
        }


class MockSentenceTransformer:
    """Mock sentence transformer for testing."""

    def __init__(self, device: str = "cpu"):
        self.device = device

    def encode(self, sentences: List[str], **kwargs) -> MockTensor:
        """Mock sentence encoding."""
        return MockTensor([len(sentences), 384])


class NeuralScorer:
    """
    Scores content using neural models for coherence and toxicity.
    Handles batching for efficient GPU utilization.
    """

    def __init__(
        self,
        coherence_model_name: str = "all-MiniLM-L6-v2",
        toxicity_model_name: str = "original",
        device: str | None = None,
    ) -> None:
        """Initializes the NeuralScorer, loading models to the appropriate device."""
        if device is None:
            self.device = "cuda" if not _TEST_MODE and torch and torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"NeuralScorer is using device: {self.device}")

        # Use mock models in test mode to avoid network calls
        if _TEST_MODE:
            logger.info("NeuralScorer running in test mode - using mock models")
            self.coherence_model = MockSentenceTransformer(device=self.device)
            self.toxicity_model = MockToxicityModel(device=self.device)
            self._is_mock = True
            return

        # Production mode - load real models
        if not HAS_ML_LIBS:
            raise ImportError("ML libraries (torch, sentence_transformers, detoxify) are required for NeuralScorer")

        if SentenceTransformer is None or Detoxify is None:
            raise ImportError("ML libraries (torch, sentence_transformers, detoxify) are required for NeuralScorer")

        self.coherence_model = SentenceTransformer(coherence_model_name, device=self.device)
        self.toxicity_model = Detoxify(toxicity_model_name, device=self.device)
        self._is_mock = False

    async def score(
        self,
        content: ExtractedContent,
        metadata: ContentMetadata,
        score: QualityScore,
    ) -> None:
        """
        Scores a single piece of content.
        This is a convenience wrapper around score_batch for protocol compliance.
        """
        if not content.text or content.word_count < 20:
            score.coherence_score = 0.0
            score.toxicity_score = 0.0
            return

        results = await self.score_batch([content])
        if results:
            res = results[0]
            score.coherence_score = res["coherence_score"]
            score.toxicity_score = res["toxicity_score"]
            score.quality_factors.update(res["quality_factors"])

    async def score_batch(self, contents: List[ExtractedContent]) -> List[Dict[str, Any]]:
        """Scores a batch of documents for coherence and toxicity."""
        texts = [content.text for content in contents if content.text and content.word_count >= 20]
        if not texts:
            return []

        # Run both models in parallel on different threads
        toxicity_task = asyncio.to_thread(self.toxicity_model.predict, texts)
        coherence_task = self._calculate_coherence_batch(texts)

        toxicity_results, coherence_scores = await asyncio.gather(
            toxicity_task,
            coherence_task,
        )

        results = []
        for i, content in enumerate(contents):
            if not content.text or content.word_count < 20:
                results.append(
                    {
                        "coherence_score": 0.0,
                        "toxicity_score": 0.0,
                        "quality_factors": {},
                    }
                )
                continue

            toxicity_score = toxicity_results["toxicity"][i]
            coherence_score = coherence_scores[i]

            results.append(
                {
                    "coherence_score": coherence_score,
                    "toxicity_score": toxicity_score,
                    "quality_factors": {
                        "neural_coherence": coherence_score,
                        "toxicity": toxicity_score,
                        **{key: value[i] for key, value in toxicity_results.items()},
                    },
                }
            )
        return results

    async def _calculate_coherence_batch(self, texts: List[str]) -> List[float]:
        """Calculates coherence for a batch of texts."""
        return await asyncio.to_thread(self._calculate_coherence_sync, texts)

    def _calculate_coherence_sync(self, texts: List[str]) -> List[float]:
        """Synchronous part of coherence calculation."""
        # This is a simplified approach. A more advanced one might split texts into sentences first.
        # For now, we embed paragraphs or chunks.
        all_sentences = [text.split(".") for text in texts]

        coherence_scores = []
        for sentences in all_sentences:
            sentences = [s for s in sentences if len(s.strip()) > 10]
            if len(sentences) < 2:
                coherence_scores.append(0.5)  # Neutral score for short texts
                continue

            embeddings = self.coherence_model.encode(
                sentences,
                convert_to_tensor=True,
                device=self.device,
                show_progress_bar=False,
            )

            if self._is_mock:
                # Mock coherence calculation - return reasonable score
                coherence_scores.append(0.7)
            else:
                # Real coherence calculation
                if cos_sim is not None and torch is not None:
                    sims = cos_sim(embeddings[:-1], embeddings[1:])
                    coherence_scores.append(torch.diag(sims).mean().item())
                else:
                    # Fallback if libraries not available
                    coherence_scores.append(0.5)

        return coherence_scores
