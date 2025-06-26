"""
Implements curriculum learning sampling strategies.
"""
from __future__ import annotations

import random
from collections import defaultdict
from typing import List, Tuple

import numpy as np

from quarrycore.config.config import SamplingConfig
from quarrycore.protocols import ContentMetadata, QualityScore


class Sampler:
    """Selects and orders documents to create a curriculum."""

    def __init__(self, config: SamplingConfig):
        self.config = config

    def sample(
        self,
        documents: List[Tuple[ContentMetadata, QualityScore]],
        target_size: int,
    ) -> List[Tuple[ContentMetadata, QualityScore]]:
        """
        Applies the configured sampling strategy to the available documents.

        Args:
            documents: A list of available documents with their metadata and scores.
            target_size: The desired number of documents in the final dataset.

        Returns:
            A new list of documents, selected and ordered for the curriculum.
        """
        if self.config.strategy == "curriculum":
            return self._curriculum_sample(documents, target_size)
        # Add other strategies like 'balanced' or 'random' here if needed
        return self._curriculum_sample(documents, target_size)

    def _curriculum_sample(
        self,
        documents: List[Tuple[ContentMetadata, QualityScore]],
        target_size: int,
    ) -> List[Tuple[ContentMetadata, QualityScore]]:
        """
        Performs stratified, quality-weighted rejection sampling, then orders by difficulty.
        """
        if not documents:
            return []

        # 1. Stratify by domain
        by_domain = defaultdict(list)
        for doc, score in documents:
            by_domain[doc.domain_type].append((doc, score))

        # 2. Perform rejection sampling within each stratum
        selected_docs = []
        domain_balance = self.config.domain_balance or {
            domain: 1.0 / len(by_domain) for domain in by_domain
        }

        for domain, proportion in domain_balance.items():
            if domain not in by_domain:
                continue

            stratum_size = int(target_size * proportion)
            stratum_candidates = by_domain[domain]
            if not stratum_candidates:
                continue
            
            # Rejection sampling
            stratum_selected = self._rejection_sample(
                stratum_candidates, stratum_size
            )
            selected_docs.extend(stratum_selected)

        # 3. Sort by difficulty (quality score)
        # Lower quality scores are considered "easier" for the curriculum
        selected_docs.sort(key=lambda item: item[1].overall_score)
        
        return selected_docs

    def _rejection_sample(
        self,
        candidates: List[Tuple[ContentMetadata, QualityScore]],
        target_size: int,
    ) -> List[Tuple[ContentMetadata, QualityScore]]:
        """
        Selects items using quality-weighted rejection sampling.
        """
        if target_size >= len(candidates):
            return candidates

        selected = []
        factor = int(self.config.rejection_sampling_factor)
        
        for _ in range(target_size):
            # Draw N candidates and pick the best one
            sample_group = random.sample(candidates, k=min(factor, len(candidates)))
            
            best_in_group = max(sample_group, key=lambda item: item[1].overall_score)
            selected.append(best_in_group)
            
            # Remove the selected item to prevent re-sampling
            # This is inefficient for large lists, but fine for this purpose
            candidates.remove(best_in_group)

        return selected 