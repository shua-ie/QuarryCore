from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    import numpy as np  # type: ignore[import-not-found]
    import torch  # type: ignore[import-not-found]
    from detoxify import Detoxify  # type: ignore[import-not-found]
    from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
    from sentence_transformers.util import cos_sim  # type: ignore[import-not-found]

from quarrycore.protocols import ContentMetadata, ExtractedContent, QualityScore

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


class NeuralScorer:
    """
    Scores content using neural models for coherence and toxicity.
    Handles batching for efficient GPU utilization.
    """

    def __init__(
        self,
        coherence_model_name: str = 'all-MiniLM-L6-v2',
        toxicity_model_name: str = 'original',
        device: str | None = None,
    ) -> None:
        """Initializes the NeuralScorer, loading models to the appropriate device."""
        if not HAS_ML_LIBS:
            raise ImportError("ML libraries (torch, sentence_transformers, detoxify) are required for NeuralScorer")
        
        if device is None:
            self.device = "cuda" if torch and torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"NeuralScorer is using device: {self.device}")

        self.coherence_model = SentenceTransformer(coherence_model_name, device=self.device)
        self.toxicity_model = Detoxify(toxicity_model_name, device=self.device)

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
                results.append({
                    "coherence_score": 0.0,
                    "toxicity_score": 0.0,
                    "quality_factors": {}
                })
                continue

            toxicity_score = toxicity_results["toxicity"][i]
            coherence_score = coherence_scores[i]

            results.append({
                "coherence_score": coherence_score,
                "toxicity_score": toxicity_score,
                "quality_factors": {
                    "neural_coherence": coherence_score,
                    "toxicity": toxicity_score,
                    **{key: value[i] for key, value in toxicity_results.items()},
                },
            })
        return results

    async def _calculate_coherence_batch(self, texts: List[str]) -> List[float]:
        """Calculates coherence for a batch of texts."""
        return await asyncio.to_thread(self._calculate_coherence_sync, texts)

    def _calculate_coherence_sync(self, texts: List[str]) -> List[float]:
        """Synchronous part of coherence calculation."""
        # This is a simplified approach. A more advanced one might split texts into sentences first.
        # For now, we embed paragraphs or chunks.
        all_sentences = [text.split('.') for text in texts]
        
        coherence_scores = []
        for sentences in all_sentences:
            sentences = [s for s in sentences if len(s.strip()) > 10]
            if len(sentences) < 2:
                coherence_scores.append(0.5) # Neutral score for short texts
                continue
            
            embeddings = self.coherence_model.encode(
                sentences,
                convert_to_tensor=True,
                device=self.device,
                show_progress_bar=False,
            )
            
            sims = cos_sim(embeddings[:-1], embeddings[1:])
            coherence_scores.append(torch.diag(sims).mean().item())
        
        return coherence_scores
