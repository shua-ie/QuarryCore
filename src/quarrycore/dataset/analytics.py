"""
Computes and reports analytics on the generated dataset.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Tuple

import numpy as np
from transformers import AutoTokenizer

from quarrycore.config.config import DatasetConfig
from quarrycore.protocols import ContentMetadata, QualityScore


class Analytics:
    """Computes and presents analytics on a final dataset."""

    def __init__(self, config: DatasetConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.chunking.tokenizer_name)

    def analyze(
        self,
        dataset: List[Dict[str, Any]],
        source_docs: List[Tuple[ContentMetadata, QualityScore]],
    ) -> Dict[str, Any]:
        """
        Performs a full analysis of the generated dataset.

        Args:
            dataset: The final list of formatted data records.
            source_docs: The original documents that were sampled to create the dataset.

        Returns:
            A dictionary containing the full analytics report.
        """
        if not dataset:
            return {"error": "Dataset is empty, no analytics to report."}

        token_counts = self._get_token_counts(dataset)
        vocab = self._get_vocabulary(dataset)

        report = {
            "general": {
                "total_records": len(dataset),
                "total_source_documents": len(source_docs),
                "vocabulary_size": len(vocab),
            },
            "token_distribution": {
                "mean": float(np.mean(token_counts)),
                "std_dev": float(np.std(token_counts)),
                "min": int(np.min(token_counts)),
                "max": int(np.max(token_counts)),
                "p25": float(np.percentile(token_counts, 25)),
                "p50": float(np.percentile(token_counts, 50)),
                "p75": float(np.percentile(token_counts, 75)),
            },
            "domain_distribution": self._get_domain_distribution(source_docs),
            "quality_distribution": self._get_quality_distribution(source_docs),
        }
        return report
    
    def _get_token_counts(self, dataset: List[Dict[str, Any]]) -> List[int]:
        """Calculates token counts for each record in the dataset."""
        # We assume the main content is in the 'text' field of the formatted record
        texts = [record.get("text", "") for record in dataset]
        return [len(tokens) for tokens in self.tokenizer(texts).input_ids]

    def _get_vocabulary(self, dataset: List[Dict[str, Any]]) -> set[str]:
        """Computes the unique vocabulary of the dataset."""
        # This is memory-intensive for large datasets.
        # A more scalable solution would use a streaming algorithm.
        vocab = set()
        texts = [record.get("text", "") for record in dataset]
        for text in texts:
            tokens = self.tokenizer.tokenize(text)
            vocab.update(tokens)
        return vocab

    def _get_domain_distribution(
        self, source_docs: List[Tuple[ContentMetadata, QualityScore]]
    ) -> Dict[str, int]:
        """Calculates the distribution of content domains."""
        counts = Counter(doc.domain_type.value for doc, score in source_docs)
        return dict(counts)
    
    def _get_quality_distribution(
        self, source_docs: List[Tuple[ContentMetadata, QualityScore]]
    ) -> Dict[str, float]:
        """Calculates the average quality scores of the source documents."""
        if not source_docs:
            return {}
        
        scores = [score.overall_score for doc, score in source_docs]
        return {
            "mean": float(np.mean(scores)),
            "std_dev": float(np.std(scores)),
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
        }

    def pretty_print_report(self, report: Dict[str, Any]) -> None:
        """Prints the analytics report in a readable format."""
        import json
        print("\n--- Dataset Analytics Report ---")
        print(json.dumps(report, indent=2))
        print("------------------------------\n") 